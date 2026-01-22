import datetime
import json
import os
import random
from parlai.utils.misc import AttrDict
import parlai.utils.logging as logging
class Conversations:
    """
    Utility class for reading and writing from ParlAI Conversations format.

    Conversations should be saved in JSONL format, where each line is
    a JSON of the following form:
    {
        'possible_conversation_level_info': True,
        'dialog':
            [   [
                    {
                        'id': 'speaker_1',
                        'text': <first utterance>,
                    },
                    {
                        'id': 'speaker_2',
                        'text': <second utterance>,
                    },
                    ...
                ],
                ...
            ]
        ...
    }
    """

    def __init__(self, datapath):
        self.conversations = self._load_conversations(datapath)
        self.metadata = self._load_metadata(datapath)

    def __len__(self):
        return len(self.conversations)

    def _load_conversations(self, datapath):
        if not os.path.isfile(datapath):
            raise RuntimeError(f'Conversations at path {datapath} not found. Double check your path.')
        conversations = []
        with open(datapath, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                conversations.append(Conversation(json.loads(line)))
        return conversations

    def _load_metadata(self, datapath):
        """
        Load metadata.

        Metadata should be saved at <identifier>.metadata
        Metadata should be of the following format:
        {
            'date': <date collected>,
            'opt': <opt used to collect the data,
            'speakers': <identity of speakers>,
            ...
            Other arguments.
        }
        """
        try:
            metadata = Metadata(datapath)
            return metadata
        except RuntimeError:
            logging.error('Metadata does not exist. Please double check your datapath.')
            return None

    def read_metadata(self):
        if self.metadata is not None:
            logging.info(self.metadata)
        else:
            logging.warn('No metadata available.')

    def __getitem__(self, index):
        return self.conversations[index]

    def __iter__(self):
        self.iterator_idx = 0
        return self

    def __next__(self):
        """
        Return the next conversation.
        """
        if self.iterator_idx >= len(self):
            raise StopIteration
        conv = self.conversations[self.iterator_idx]
        self.iterator_idx += 1
        return conv

    def read_conv_idx(self, idx):
        convo = self.conversations[idx]
        logging.info(convo)

    def read_rand_conv(self):
        idx = random.choice(range(len(self)))
        self.read_conv_idx(idx)

    @staticmethod
    def _get_path(datapath):
        fle, _ = os.path.splitext(datapath)
        return fle + '.jsonl'

    @classmethod
    def save_conversations(cls, act_list, datapath, opt, save_keys='all', context_ids='context', self_chat=False, **kwargs):
        """
        Write Conversations to file from an act list.

        Conversations assume the act list is of the following form: a list of episodes,
        each of which is comprised of a list of act pairs (i.e. a list dictionaries
        returned from one parley)
        """
        to_save = cls._get_path(datapath)
        context_ids = context_ids.split(',')
        speakers = []
        with open(to_save, 'w') as f:
            for ep in act_list:
                if not ep:
                    continue
                convo = {'dialog': [], 'context': [], 'metadata_path': Metadata._get_path(to_save)}
                for act_pair in ep:
                    new_pair = []
                    for ex in act_pair:
                        ex_id = ex.get('id')
                        if ex_id in context_ids:
                            context = True
                        else:
                            context = False
                            if ex_id not in speakers:
                                speakers.append(ex_id)
                        turn = {}
                        if save_keys != 'all':
                            save_keys_lst = save_keys.split(',')
                        else:
                            save_keys_lst = [key for key in ex.keys() if key != 'metrics']
                        for key in save_keys_lst:
                            turn[key] = ex.get(key, '')
                        turn['id'] = ex_id
                        if not context:
                            new_pair.append(turn)
                        else:
                            convo['context'].append(turn)
                    if new_pair:
                        convo['dialog'].append(new_pair)
                json_convo = json.dumps(convo)
                f.write(json_convo + '\n')
        logging.info(f'Conversations saved to file: {to_save}')
        Metadata.save_metadata(to_save, opt, self_chat=self_chat, speakers=speakers, **kwargs)