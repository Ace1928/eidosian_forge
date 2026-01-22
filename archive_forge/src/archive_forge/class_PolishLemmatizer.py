from typing import Dict, List, Tuple
from ...pipeline import Lemmatizer
from ...tokens import Token
class PolishLemmatizer(Lemmatizer):

    @classmethod
    def get_lookups_config(cls, mode: str) -> Tuple[List[str], List[str]]:
        if mode == 'pos_lookup':
            required = ['lemma_lookup_adj', 'lemma_lookup_adp', 'lemma_lookup_adv', 'lemma_lookup_aux', 'lemma_lookup_noun', 'lemma_lookup_num', 'lemma_lookup_part', 'lemma_lookup_pron', 'lemma_lookup_verb']
            return (required, [])
        else:
            return super().get_lookups_config(mode)

    def pos_lookup_lemmatize(self, token: Token) -> List[str]:
        string = token.text
        univ_pos = token.pos_
        morphology = token.morph.to_dict()
        lookup_pos = univ_pos.lower()
        if univ_pos == 'PROPN':
            lookup_pos = 'noun'
        lookup_table = self.lookups.get_table('lemma_lookup_' + lookup_pos, {})
        if univ_pos == 'NOUN':
            return self.lemmatize_noun(string, morphology, lookup_table)
        if univ_pos != 'PROPN':
            string = string.lower()
        if univ_pos == 'ADJ':
            return self.lemmatize_adj(string, morphology, lookup_table)
        elif univ_pos == 'VERB':
            return self.lemmatize_verb(string, morphology, lookup_table)
        return [lookup_table.get(string, string.lower())]

    def lemmatize_adj(self, string: str, morphology: dict, lookup_table: Dict[str, str]) -> List[str]:
        if string[:3] == 'nie':
            search_string = string[3:]
            if search_string[:3] == 'naj':
                naj_search_string = search_string[3:]
                if naj_search_string in lookup_table:
                    return [lookup_table[naj_search_string]]
            if search_string in lookup_table:
                return [lookup_table[search_string]]
        if string[:3] == 'naj':
            naj_search_string = string[3:]
            if naj_search_string in lookup_table:
                return [lookup_table[naj_search_string]]
        return [lookup_table.get(string, string)]

    def lemmatize_verb(self, string: str, morphology: dict, lookup_table: Dict[str, str]) -> List[str]:
        if string[:3] == 'nie':
            search_string = string[3:]
            if search_string in lookup_table:
                return [lookup_table[search_string]]
        return [lookup_table.get(string, string)]

    def lemmatize_noun(self, string: str, morphology: dict, lookup_table: Dict[str, str]) -> List[str]:
        if string != string.lower():
            if string.lower() in lookup_table:
                return [lookup_table[string.lower()]]
            elif string in lookup_table:
                return [lookup_table[string]]
            return [string.lower()]
        return [lookup_table.get(string, string)]