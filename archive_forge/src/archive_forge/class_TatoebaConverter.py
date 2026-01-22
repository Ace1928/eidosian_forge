import argparse
import datetime
import json
import os
import re
from pathlib import Path
from typing import Tuple
import yaml
from tqdm import tqdm
from transformers.models.marian.convert_marian_to_pytorch import (
class TatoebaConverter:
    """
    Convert Tatoeba-Challenge models to huggingface format.

    Steps:

        1. Convert numpy state dict to hf format (same code as OPUS-MT-Train conversion).
        2. Rename opus model to huggingface format. This means replace each alpha3 code with an alpha2 code if a unique
           one exists. e.g. aav-eng -> aav-en, heb-eng -> he-en
        3. Select the best model for a particular pair, parse the yml for it and write a model card. By default the
           best model is the one listed first in released-model-results, but it's also possible to specify the most
           recent one.
    """

    def __init__(self, save_dir='marian_converted'):
        assert Path(DEFAULT_REPO).exists(), 'need git clone git@github.com:Helsinki-NLP/Tatoeba-Challenge.git'
        self.download_lang_info()
        self.model_results = json.load(open('Tatoeba-Challenge/models/released-model-results.json'))
        self.alpha3_to_alpha2 = {}
        for line in open(ISO_PATH):
            parts = line.split('\t')
            if len(parts[0]) == 3 and len(parts[3]) == 2:
                self.alpha3_to_alpha2[parts[0]] = parts[3]
        for line in LANG_CODE_PATH:
            parts = line.split(',')
            if len(parts[0]) == 3 and len(parts[1]) == 2:
                self.alpha3_to_alpha2[parts[0]] = parts[1]
        self.model_card_dir = Path(save_dir)
        self.tag2name = {}
        for key, value in GROUP_MEMBERS.items():
            self.tag2name[key] = value[0]

    def convert_models(self, tatoeba_ids, dry_run=False):
        models_to_convert = [self.parse_metadata(x) for x in tatoeba_ids]
        save_dir = Path('marian_ckpt')
        dest_dir = Path(self.model_card_dir)
        dest_dir.mkdir(exist_ok=True)
        for model in tqdm(models_to_convert):
            if 'SentencePiece' not in model['pre-processing']:
                print(f"Skipping {model['release']} because it doesn't appear to use SentencePiece")
                continue
            if not os.path.exists(save_dir / model['_name']):
                download_and_unzip(f'{TATOEBA_MODELS_URL}/{model['release']}', save_dir / model['_name'])
            opus_language_groups_to_hf = convert_opus_name_to_hf_name
            pair_name = opus_language_groups_to_hf(model['_name'])
            convert(save_dir / model['_name'], dest_dir / f'opus-mt-{pair_name}')
            self.write_model_card(model, dry_run=dry_run)

    def expand_group_to_two_letter_codes(self, grp_name):
        return [self.alpha3_to_alpha2.get(x, x) for x in GROUP_MEMBERS[grp_name][1]]

    def is_group(self, code, name):
        return 'languages' in name or len(GROUP_MEMBERS.get(code, [])) > 1

    def get_tags(self, code, name):
        if len(code) == 2:
            assert 'languages' not in name, f'{code}: {name}'
            return [code]
        elif self.is_group(code, name):
            group = self.expand_group_to_two_letter_codes(code)
            group.append(code)
            return group
        else:
            print(f'Three letter monolingual code: {code}')
            return [code]

    def resolve_lang_code(self, src, tgt) -> Tuple[str, str]:
        src_tags = self.get_tags(src, self.tag2name[src])
        tgt_tags = self.get_tags(tgt, self.tag2name[tgt])
        return (src_tags, tgt_tags)

    @staticmethod
    def model_type_info_from_model_name(name):
        info = {'_has_backtranslated_data': False}
        if '1m' in name:
            info['_data_per_pair'] = str(1000000.0)
        if '2m' in name:
            info['_data_per_pair'] = str(2000000.0)
        if '4m' in name:
            info['_data_per_pair'] = str(4000000.0)
        if '+bt' in name:
            info['_has_backtranslated_data'] = True
        if 'tuned4' in name:
            info['_tuned'] = re.search('tuned4[^-]+', name).group()
        return info

    def write_model_card(self, model_dict, dry_run=False) -> str:
        """
        Construct card from data parsed from YAML and the model's name. upload command: aws s3 sync model_card_dir
        s3://models.huggingface.co/bert/Helsinki-NLP/ --dryrun
        """
        model_dir_url = f'{TATOEBA_MODELS_URL}/{model_dict['release']}'
        long_pair = model_dict['_name'].split('-')
        assert len(long_pair) == 2, f"got a translation pair {model_dict['_name']} that doesn't appear to be a pair"
        short_src = self.alpha3_to_alpha2.get(long_pair[0], long_pair[0])
        short_tgt = self.alpha3_to_alpha2.get(long_pair[1], long_pair[1])
        model_dict['_hf_model_id'] = f'opus-mt-{short_src}-{short_tgt}'
        a3_src, a3_tgt = model_dict['_name'].split('-')
        resolved_src_tags, resolved_tgt_tags = self.resolve_lang_code(a3_src, a3_tgt)
        a2_src_tags, a2_tgt_tags = ([], [])
        for tag in resolved_src_tags:
            if tag not in self.alpha3_to_alpha2:
                a2_src_tags.append(tag)
        for tag in resolved_tgt_tags:
            if tag not in self.alpha3_to_alpha2:
                a2_tgt_tags.append(tag)
        lang_tags = dedup(a2_src_tags + a2_tgt_tags)
        src_multilingual, tgt_multilingual = (len(a2_src_tags) > 1, len(a2_tgt_tags) > 1)
        s, t = (','.join(a2_src_tags), ','.join(a2_tgt_tags))
        metadata = {'hf_name': model_dict['_name'], 'source_languages': s, 'target_languages': t, 'opus_readme_url': f'{model_dir_url}/README.md', 'original_repo': 'Tatoeba-Challenge', 'tags': ['translation'], 'languages': lang_tags}
        lang_tags = l2front_matter(lang_tags)
        metadata['src_constituents'] = list(GROUP_MEMBERS[a3_src][1])
        metadata['tgt_constituents'] = list(GROUP_MEMBERS[a3_tgt][1])
        metadata['src_multilingual'] = src_multilingual
        metadata['tgt_multilingual'] = tgt_multilingual
        backtranslated_data = ''
        if model_dict['_has_backtranslated_data']:
            backtranslated_data = ' with backtranslations'
        multilingual_data = ''
        if '_data_per_pair' in model_dict:
            multilingual_data = f'* data per pair in multilingual model: {model_dict['_data_per_pair']}\n'
        tuned = ''
        if '_tuned' in model_dict:
            tuned = f'* multilingual model tuned for: {model_dict['_tuned']}\n'
        model_base_filename = model_dict['release'].split('/')[-1]
        download = f'* download original weights: [{model_base_filename}]({model_dir_url}/{model_dict['release']})\n'
        langtoken = ''
        if tgt_multilingual:
            langtoken = '* a sentence-initial language token is required in the form of >>id<<(id = valid, usually three-letter target language ID)\n'
        metadata.update(get_system_metadata(DEFAULT_REPO))
        scorestable = ''
        for k, v in model_dict.items():
            if 'scores' in k:
                this_score_table = f'* {k}\n|Test set|score|\n|---|---|\n'
                pairs = sorted(v.items(), key=lambda x: x[1], reverse=True)
                for pair in pairs:
                    this_score_table += f'|{pair[0]}|{pair[1]}|\n'
                scorestable += this_score_table
        datainfo = ''
        if 'training-data' in model_dict:
            datainfo += '* Training data: \n'
            for k, v in model_dict['training-data'].items():
                datainfo += f'  * {str(k)}: {str(v)}\n'
        if 'validation-data' in model_dict:
            datainfo += '* Validation data: \n'
            for k, v in model_dict['validation-data'].items():
                datainfo += f'  * {str(k)}: {str(v)}\n'
        if 'test-data' in model_dict:
            datainfo += '* Test data: \n'
            for k, v in model_dict['test-data'].items():
                datainfo += f'  * {str(k)}: {str(v)}\n'
        testsetfilename = model_dict['release'].replace('.zip', '.test.txt')
        testscoresfilename = model_dict['release'].replace('.zip', '.eval.txt')
        testset = f'* test set translations file: [test.txt]({model_dir_url}/{testsetfilename})\n'
        testscores = f'* test set scores file: [eval.txt]({model_dir_url}/{testscoresfilename})\n'
        readme_url = f'{TATOEBA_MODELS_URL}/{model_dict['_name']}/README.md'
        extra_markdown = f'\n### {model_dict['_name']}\n\n* source language name: {self.tag2name[a3_src]}\n* target language name: {self.tag2name[a3_tgt]}\n* OPUS readme: [README.md]({readme_url})\n'
        content = f'\n* model: {model_dict['modeltype']}\n* source language code{src_multilingual * 's'}: {', '.join(a2_src_tags)}\n* target language code{tgt_multilingual * 's'}: {', '.join(a2_tgt_tags)}\n* dataset: opus {backtranslated_data}\n* release date: {model_dict['release-date']}\n* pre-processing: {model_dict['pre-processing']}\n' + multilingual_data + tuned + download + langtoken + datainfo + testset + testscores + scorestable
        content = FRONT_MATTER_TEMPLATE.format(lang_tags) + extra_markdown + content
        items = '\n'.join([f'* {k}: {v}' for k, v in metadata.items()])
        sec3 = '\n### System Info: \n' + items
        content += sec3
        if dry_run:
            print('CONTENT:')
            print(content)
            print('METADATA:')
            print(metadata)
            return
        sub_dir = self.model_card_dir / model_dict['_hf_model_id']
        sub_dir.mkdir(exist_ok=True)
        dest = sub_dir / 'README.md'
        dest.open('w').write(content)
        for k, v in metadata.items():
            if isinstance(v, datetime.date):
                metadata[k] = datetime.datetime.strftime(v, '%Y-%m-%d')
        with open(sub_dir / 'metadata.json', 'w', encoding='utf-8') as writeobj:
            json.dump(metadata, writeobj)

    def download_lang_info(self):
        Path(LANG_CODE_PATH).parent.mkdir(exist_ok=True)
        import wget
        if not os.path.exists(ISO_PATH):
            wget.download(ISO_URL, ISO_PATH)
        if not os.path.exists(LANG_CODE_PATH):
            wget.download(LANG_CODE_URL, LANG_CODE_PATH)

    def parse_metadata(self, model_name, repo_path=DEFAULT_MODEL_DIR, method='best'):
        p = Path(repo_path) / model_name

        def url_to_name(url):
            return url.split('/')[-1].split('.')[0]
        if model_name not in self.model_results:
            method = 'newest'
        if method == 'best':
            results = [url_to_name(model['download']) for model in self.model_results[model_name]]
            ymls = [f for f in os.listdir(p) if f.endswith('.yml') and f[:-4] in results]
            ymls.sort(key=lambda x: results.index(x[:-4]))
            metadata = yaml.safe_load(open(p / ymls[0]))
            metadata.update(self.model_type_info_from_model_name(ymls[0][:-4]))
        elif method == 'newest':
            ymls = [f for f in os.listdir(p) if f.endswith('.yml')]
            ymls.sort(key=lambda x: datetime.datetime.strptime(re.search('\\d\\d\\d\\d-\\d\\d?-\\d\\d?', x).group(), '%Y-%m-%d'))
            metadata = yaml.safe_load(open(p / ymls[-1]))
            metadata.update(self.model_type_info_from_model_name(ymls[-1][:-4]))
        else:
            raise NotImplementedError(f"Don't know argument method='{method}' to parse_metadata()")
        metadata['_name'] = model_name
        return metadata