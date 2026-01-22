import ast
import collections
import contextlib
import functools
import itertools
import re
from numbers import Number
from typing import (
from more_itertools import windowed_complete
from typeguard import typechecked
from typing_extensions import Annotated, Literal
class engine:

    def __init__(self) -> None:
        self.classical_dict = def_classical.copy()
        self.persistent_count: Optional[int] = None
        self.mill_count = 0
        self.pl_sb_user_defined: List[Optional[Word]] = []
        self.pl_v_user_defined: List[Optional[Word]] = []
        self.pl_adj_user_defined: List[Optional[Word]] = []
        self.si_sb_user_defined: List[Optional[Word]] = []
        self.A_a_user_defined: List[Optional[Word]] = []
        self.thegender = 'neuter'
        self.__number_args: Optional[Dict[str, str]] = None

    @property
    def _number_args(self):
        return cast(Dict[str, str], self.__number_args)

    @_number_args.setter
    def _number_args(self, val):
        self.__number_args = val

    @typechecked
    def defnoun(self, singular: Optional[Word], plural: Optional[Word]) -> int:
        """
        Set the noun plural of singular to plural.

        """
        self.checkpat(singular)
        self.checkpatplural(plural)
        self.pl_sb_user_defined.extend((singular, plural))
        self.si_sb_user_defined.extend((plural, singular))
        return 1

    @typechecked
    def defverb(self, s1: Optional[Word], p1: Optional[Word], s2: Optional[Word], p2: Optional[Word], s3: Optional[Word], p3: Optional[Word]) -> int:
        """
        Set the verb plurals for s1, s2 and s3 to p1, p2 and p3 respectively.

        Where 1, 2 and 3 represent the 1st, 2nd and 3rd person forms of the verb.

        """
        self.checkpat(s1)
        self.checkpat(s2)
        self.checkpat(s3)
        self.checkpatplural(p1)
        self.checkpatplural(p2)
        self.checkpatplural(p3)
        self.pl_v_user_defined.extend((s1, p1, s2, p2, s3, p3))
        return 1

    @typechecked
    def defadj(self, singular: Optional[Word], plural: Optional[Word]) -> int:
        """
        Set the adjective plural of singular to plural.

        """
        self.checkpat(singular)
        self.checkpatplural(plural)
        self.pl_adj_user_defined.extend((singular, plural))
        return 1

    @typechecked
    def defa(self, pattern: Optional[Word]) -> int:
        """
        Define the indefinite article as 'a' for words matching pattern.

        """
        self.checkpat(pattern)
        self.A_a_user_defined.extend((pattern, 'a'))
        return 1

    @typechecked
    def defan(self, pattern: Optional[Word]) -> int:
        """
        Define the indefinite article as 'an' for words matching pattern.

        """
        self.checkpat(pattern)
        self.A_a_user_defined.extend((pattern, 'an'))
        return 1

    def checkpat(self, pattern: Optional[Word]) -> None:
        """
        check for errors in a regex pattern
        """
        if pattern is None:
            return
        try:
            re.match(pattern, '')
        except re.error as err:
            raise BadUserDefinedPatternError(pattern) from err

    def checkpatplural(self, pattern: Optional[Word]) -> None:
        """
        check for errors in a regex replace pattern
        """
        return

    @typechecked
    def ud_match(self, word: Word, wordlist: Sequence[Optional[Word]]) -> Optional[str]:
        for i in range(len(wordlist) - 2, -2, -2):
            mo = re.search(f'^{wordlist[i]}$', word, re.IGNORECASE)
            if mo:
                if wordlist[i + 1] is None:
                    return None
                pl = DOLLAR_DIGITS.sub('\\\\1', cast(Word, wordlist[i + 1]))
                return mo.expand(pl)
        return None

    def classical(self, **kwargs) -> None:
        """
        turn classical mode on and off for various categories

        turn on all classical modes:
        classical()
        classical(all=True)

        turn on or off specific claassical modes:
        e.g.
        classical(herd=True)
        classical(names=False)

        By default all classical modes are off except names.

        unknown value in args or key in kwargs raises
        exception: UnknownClasicalModeError

        """
        if not kwargs:
            self.classical_dict = all_classical.copy()
            return
        if 'all' in kwargs:
            if kwargs['all']:
                self.classical_dict = all_classical.copy()
            else:
                self.classical_dict = no_classical.copy()
        for k, v in kwargs.items():
            if k in def_classical:
                self.classical_dict[k] = v
            else:
                raise UnknownClassicalModeError

    def num(self, count: Optional[int]=None, show: Optional[int]=None) -> str:
        """
        Set the number to be used in other method calls.

        Returns count.

        Set show to False to return '' instead.

        """
        if count is not None:
            try:
                self.persistent_count = int(count)
            except ValueError as err:
                raise BadNumValueError from err
            if show is None or show:
                return str(count)
        else:
            self.persistent_count = None
        return ''

    def gender(self, gender: str) -> None:
        """
        set the gender for the singular of plural pronouns

        can be one of:
        'neuter'                ('they' -> 'it')
        'feminine'              ('they' -> 'she')
        'masculine'             ('they' -> 'he')
        'gender-neutral'        ('they' -> 'they')
        'feminine or masculine' ('they' -> 'she or he')
        'masculine or feminine' ('they' -> 'he or she')
        """
        if gender in singular_pronoun_genders:
            self.thegender = gender
        else:
            raise BadGenderError

    def _get_value_from_ast(self, obj):
        """
        Return the value of the ast object.
        """
        if isinstance(obj, ast.Num):
            return obj.n
        elif isinstance(obj, ast.Str):
            return obj.s
        elif isinstance(obj, ast.List):
            return [self._get_value_from_ast(e) for e in obj.elts]
        elif isinstance(obj, ast.Tuple):
            return tuple([self._get_value_from_ast(e) for e in obj.elts])
        elif isinstance(obj, ast.NameConstant):
            return obj.value
        raise NameError(f"name '{obj.id}' is not defined")

    def _string_to_substitute(self, mo: Match, methods_dict: Dict[str, Callable]) -> str:
        """
        Return the string to be substituted for the match.
        """
        matched_text, f_name = mo.groups()
        if f_name not in methods_dict:
            return matched_text
        a_tree = ast.parse(matched_text)
        args_list = [self._get_value_from_ast(a) for a in a_tree.body[0].value.args]
        kwargs_list = {kw.arg: self._get_value_from_ast(kw.value) for kw in a_tree.body[0].value.keywords}
        return methods_dict[f_name](*args_list, **kwargs_list)

    @typechecked
    def inflect(self, text: Word) -> str:
        """
        Perform inflections in a string.

        e.g. inflect('The plural of cat is plural(cat)') returns
        'The plural of cat is cats'

        can use plural, plural_noun, plural_verb, plural_adj,
        singular_noun, a, an, no, ordinal, number_to_words,
        and prespart

        """
        save_persistent_count = self.persistent_count
        methods_dict: Dict[str, Callable] = {'plural': self.plural, 'plural_adj': self.plural_adj, 'plural_noun': self.plural_noun, 'plural_verb': self.plural_verb, 'singular_noun': self.singular_noun, 'a': self.a, 'an': self.a, 'no': self.no, 'ordinal': self.ordinal, 'number_to_words': self.number_to_words, 'present_participle': self.present_participle, 'num': self.num}
        output = FUNCTION_CALL.sub(lambda mo: self._string_to_substitute(mo, methods_dict), text)
        self.persistent_count = save_persistent_count
        return output

    def postprocess(self, orig: str, inflected) -> str:
        inflected = str(inflected)
        if '|' in inflected:
            word_options = inflected.split('|')
            if len(word_options[0].split(' ')) == len(word_options[1].split(' ')):
                result = inflected.split('|')[self.classical_dict['all']].split(' ')
            else:
                result = inflected.split(' ')
                for index, word in enumerate(result):
                    if '|' in word:
                        result[index] = word.split('|')[self.classical_dict['all']]
        else:
            result = inflected.split(' ')
        for index, word in enumerate(orig.split(' ')):
            if word == 'I':
                continue
            if word.capitalize() == word:
                result[index] = result[index].capitalize()
            if word == word.upper():
                result[index] = result[index].upper()
        return ' '.join(result)

    def partition_word(self, text: str) -> Tuple[str, str, str]:
        mo = PARTITION_WORD.search(text)
        if mo:
            return (mo.group(1), mo.group(2), mo.group(3))
        else:
            return ('', '', '')

    @typechecked
    def plural(self, text: Word, count: Optional[Union[str, int, Any]]=None) -> str:
        """
        Return the plural of text.

        If count supplied, then return text if count is one of:
            1, a, an, one, each, every, this, that

        otherwise return the plural.

        Whitespace at the start and end is preserved.

        """
        pre, word, post = self.partition_word(text)
        if not word:
            return text
        plural = self.postprocess(word, self._pl_special_adjective(word, count) or self._pl_special_verb(word, count) or self._plnoun(word, count))
        return f'{pre}{plural}{post}'

    @typechecked
    def plural_noun(self, text: Word, count: Optional[Union[str, int, Any]]=None) -> str:
        """
        Return the plural of text, where text is a noun.

        If count supplied, then return text if count is one of:
            1, a, an, one, each, every, this, that

        otherwise return the plural.

        Whitespace at the start and end is preserved.

        """
        pre, word, post = self.partition_word(text)
        if not word:
            return text
        plural = self.postprocess(word, self._plnoun(word, count))
        return f'{pre}{plural}{post}'

    @typechecked
    def plural_verb(self, text: Word, count: Optional[Union[str, int, Any]]=None) -> str:
        """
        Return the plural of text, where text is a verb.

        If count supplied, then return text if count is one of:
            1, a, an, one, each, every, this, that

        otherwise return the plural.

        Whitespace at the start and end is preserved.

        """
        pre, word, post = self.partition_word(text)
        if not word:
            return text
        plural = self.postprocess(word, self._pl_special_verb(word, count) or self._pl_general_verb(word, count))
        return f'{pre}{plural}{post}'

    @typechecked
    def plural_adj(self, text: Word, count: Optional[Union[str, int, Any]]=None) -> str:
        """
        Return the plural of text, where text is an adjective.

        If count supplied, then return text if count is one of:
            1, a, an, one, each, every, this, that

        otherwise return the plural.

        Whitespace at the start and end is preserved.

        """
        pre, word, post = self.partition_word(text)
        if not word:
            return text
        plural = self.postprocess(word, self._pl_special_adjective(word, count) or word)
        return f'{pre}{plural}{post}'

    @typechecked
    def compare(self, word1: Word, word2: Word) -> Union[str, bool]:
        """
        compare word1 and word2 for equality regardless of plurality

        return values:
        eq - the strings are equal
        p:s - word1 is the plural of word2
        s:p - word2 is the plural of word1
        p:p - word1 and word2 are two different plural forms of the one word
        False - otherwise

        >>> compare = engine().compare
        >>> compare("egg", "eggs")
        's:p'
        >>> compare('egg', 'egg')
        'eq'

        Words should not be empty.

        >>> compare('egg', '')
        Traceback (most recent call last):
        ...
        typeguard.TypeCheckError:...is not an instance of inflect.Word
        """
        norms = (self.plural_noun, self.plural_verb, self.plural_adj)
        results = (self._plequal(word1, word2, norm) for norm in norms)
        return next(filter(None, results), False)

    @typechecked
    def compare_nouns(self, word1: Word, word2: Word) -> Union[str, bool]:
        """
        compare word1 and word2 for equality regardless of plurality
        word1 and word2 are to be treated as nouns

        return values:
        eq - the strings are equal
        p:s - word1 is the plural of word2
        s:p - word2 is the plural of word1
        p:p - word1 and word2 are two different plural forms of the one word
        False - otherwise

        """
        return self._plequal(word1, word2, self.plural_noun)

    @typechecked
    def compare_verbs(self, word1: Word, word2: Word) -> Union[str, bool]:
        """
        compare word1 and word2 for equality regardless of plurality
        word1 and word2 are to be treated as verbs

        return values:
        eq - the strings are equal
        p:s - word1 is the plural of word2
        s:p - word2 is the plural of word1
        p:p - word1 and word2 are two different plural forms of the one word
        False - otherwise

        """
        return self._plequal(word1, word2, self.plural_verb)

    @typechecked
    def compare_adjs(self, word1: Word, word2: Word) -> Union[str, bool]:
        """
        compare word1 and word2 for equality regardless of plurality
        word1 and word2 are to be treated as adjectives

        return values:
        eq - the strings are equal
        p:s - word1 is the plural of word2
        s:p - word2 is the plural of word1
        p:p - word1 and word2 are two different plural forms of the one word
        False - otherwise

        """
        return self._plequal(word1, word2, self.plural_adj)

    @typechecked
    def singular_noun(self, text: Word, count: Optional[Union[int, str, Any]]=None, gender: Optional[str]=None) -> Union[str, Literal[False]]:
        """
        Return the singular of text, where text is a plural noun.

        If count supplied, then return the singular if count is one of:
            1, a, an, one, each, every, this, that or if count is None

        otherwise return text unchanged.

        Whitespace at the start and end is preserved.

        >>> p = engine()
        >>> p.singular_noun('horses')
        'horse'
        >>> p.singular_noun('knights')
        'knight'

        Returns False when a singular noun is passed.

        >>> p.singular_noun('horse')
        False
        >>> p.singular_noun('knight')
        False
        >>> p.singular_noun('soldier')
        False

        """
        pre, word, post = self.partition_word(text)
        if not word:
            return text
        sing = self._sinoun(word, count=count, gender=gender)
        if sing is not False:
            plural = self.postprocess(word, sing)
            return f'{pre}{plural}{post}'
        return False

    def _plequal(self, word1: str, word2: str, pl) -> Union[str, bool]:
        classval = self.classical_dict.copy()
        self.classical_dict = all_classical.copy()
        if word1 == word2:
            return 'eq'
        if word1 == pl(word2):
            return 'p:s'
        if pl(word1) == word2:
            return 's:p'
        self.classical_dict = no_classical.copy()
        if word1 == pl(word2):
            return 'p:s'
        if pl(word1) == word2:
            return 's:p'
        self.classical_dict = classval.copy()
        if pl == self.plural or pl == self.plural_noun:
            if self._pl_check_plurals_N(word1, word2):
                return 'p:p'
            if self._pl_check_plurals_N(word2, word1):
                return 'p:p'
        if pl == self.plural or pl == self.plural_adj:
            if self._pl_check_plurals_adj(word1, word2):
                return 'p:p'
        return False

    def _pl_reg_plurals(self, pair: str, stems: str, end1: str, end2: str) -> bool:
        pattern = f'({stems})({end1}\\|\\1{end2}|{end2}\\|\\1{end1})'
        return bool(re.search(pattern, pair))

    def _pl_check_plurals_N(self, word1: str, word2: str) -> bool:
        stem_endings = ((pl_sb_C_a_ata, 'as', 'ata'), (pl_sb_C_is_ides, 'is', 'ides'), (pl_sb_C_a_ae, 's', 'e'), (pl_sb_C_en_ina, 'ens', 'ina'), (pl_sb_C_um_a, 'ums', 'a'), (pl_sb_C_us_i, 'uses', 'i'), (pl_sb_C_on_a, 'ons', 'a'), (pl_sb_C_o_i_stems, 'os', 'i'), (pl_sb_C_ex_ices, 'exes', 'ices'), (pl_sb_C_ix_ices, 'ixes', 'ices'), (pl_sb_C_i, 's', 'i'), (pl_sb_C_im, 's', 'im'), ('.*eau', 's', 'x'), ('.*ieu', 's', 'x'), ('.*tri', 'xes', 'ces'), ('.{2,}[yia]n', 'xes', 'ges'))
        words = map(Words, (word1, word2))
        pair = '|'.join((word.last for word in words))
        return pair in pl_sb_irregular_s.values() or pair in pl_sb_irregular.values() or pair in pl_sb_irregular_caps.values() or any((self._pl_reg_plurals(pair, stems, end1, end2) for stems, end1, end2 in stem_endings))

    def _pl_check_plurals_adj(self, word1: str, word2: str) -> bool:
        word1a = word1[:word1.rfind("'")] if word1.endswith(("'s", "'")) else ''
        word2a = word2[:word2.rfind("'")] if word2.endswith(("'s", "'")) else ''
        return bool(word1a) and bool(word2a) and (self._pl_check_plurals_N(word1a, word2a) or self._pl_check_plurals_N(word2a, word1a))

    def get_count(self, count: Optional[Union[str, int]]=None) -> Union[str, int]:
        if count is None and self.persistent_count is not None:
            count = self.persistent_count
        if count is not None:
            count = 1 if str(count) in pl_count_one or (self.classical_dict['zero'] and str(count).lower() in pl_count_zero) else 2
        else:
            count = ''
        return count

    def _plnoun(self, word: str, count: Optional[Union[str, int]]=None) -> str:
        count = self.get_count(count)
        if count == 1:
            return word
        value = self.ud_match(word, self.pl_sb_user_defined)
        if value is not None:
            return value
        if word == '':
            return word
        word = Words(word)
        if word.last.lower() in pl_sb_uninflected_complete:
            if len(word.split_) >= 3:
                return self._handle_long_compounds(word, count=2) or word
            return word
        if word in pl_sb_uninflected_caps:
            return word
        for k, v in pl_sb_uninflected_bysize.items():
            if word.lowered[-k:] in v:
                return word
        if self.classical_dict['herd'] and word.last.lower() in pl_sb_uninflected_herd:
            return word
        mo = PL_SB_POSTFIX_ADJ_STEMS_RE.search(word)
        if mo and mo.group(2) != '':
            return f'{self._plnoun(mo.group(1), 2)}{mo.group(2)}'
        if ' a ' in word.lowered or '-a-' in word.lowered:
            mo = PL_SB_PREP_DUAL_COMPOUND_RE.search(word)
            if mo and mo.group(2) != '' and (mo.group(3) != ''):
                return f'{self._plnoun(mo.group(1), 2)}{mo.group(2)}{self._plnoun(mo.group(3))}'
        if len(word.split_) >= 3:
            handled_words = self._handle_long_compounds(word, count=2)
            if handled_words is not None:
                return handled_words
        mo = DENOMINATOR.search(word.lowered)
        if mo:
            index = len(mo.group('denominator'))
            return f'{self._plnoun(word[:index])}{word[index:]}'
        if len(word.split_) >= 2 and word.split_[-2] == 'degree':
            return ' '.join([self._plnoun(word.first)] + word.split_[1:])
        with contextlib.suppress(ValueError):
            return self._handle_prepositional_phrase(word.lowered, functools.partial(self._plnoun, count=2), '-')
        for k, v in pl_pron_acc_keys_bysize.items():
            if word.lowered[-k:] in v:
                for pk, pv in pl_prep_bysize.items():
                    if word.lowered[:pk] in pv:
                        if word.lowered.split() == [word.lowered[:pk], word.lowered[-k:]]:
                            return word.lowered[:-k] + pl_pron_acc[word.lowered[-k:]]
        try:
            return pl_pron_nom[word.lowered]
        except KeyError:
            pass
        try:
            return pl_pron_acc[word.lowered]
        except KeyError:
            pass
        if word.last in pl_sb_irregular_caps:
            llen = len(word.last)
            return f'{word[:-llen]}{pl_sb_irregular_caps[word.last]}'
        lowered_last = word.last.lower()
        if lowered_last in pl_sb_irregular:
            llen = len(lowered_last)
            return f'{word[:-llen]}{pl_sb_irregular[lowered_last]}'
        dash_split = word.lowered.split('-')
        if ' '.join(dash_split[-2:]).lower() in pl_sb_irregular_compound:
            llen = len(' '.join(dash_split[-2:]))
            return f'{word[:-llen]}{pl_sb_irregular_compound[' '.join(dash_split[-2:]).lower()]}'
        if word.lowered[-3:] == 'quy':
            return f'{word[:-1]}ies'
        if word.lowered[-6:] == 'person':
            if self.classical_dict['persons']:
                return f'{word}s'
            else:
                return f'{word[:-4]}ople'
        if word.lowered[-3:] == 'man':
            for k, v in pl_sb_U_man_mans_bysize.items():
                if word.lowered[-k:] in v:
                    return f'{word}s'
            for k, v in pl_sb_U_man_mans_caps_bysize.items():
                if word[-k:] in v:
                    return f'{word}s'
            return f'{word[:-3]}men'
        if word.lowered[-5:] == 'mouse':
            return f'{word[:-5]}mice'
        if word.lowered[-5:] == 'louse':
            v = pl_sb_U_louse_lice_bysize.get(len(word))
            if v and word.lowered in v:
                return f'{word[:-5]}lice'
            return f'{word}s'
        if word.lowered[-5:] == 'goose':
            return f'{word[:-5]}geese'
        if word.lowered[-5:] == 'tooth':
            return f'{word[:-5]}teeth'
        if word.lowered[-4:] == 'foot':
            return f'{word[:-4]}feet'
        if word.lowered[-4:] == 'taco':
            return f'{word[:-5]}tacos'
        if word.lowered == 'die':
            return 'dice'
        if word.lowered[-4:] == 'ceps':
            return word
        if word.lowered[-4:] == 'zoon':
            return f'{word[:-2]}a'
        if word.lowered[-3:] in ('cis', 'sis', 'xis'):
            return f'{word[:-2]}es'
        for lastlet, d, numend, post in (('h', pl_sb_U_ch_chs_bysize, None, 's'), ('x', pl_sb_U_ex_ices_bysize, -2, 'ices'), ('x', pl_sb_U_ix_ices_bysize, -2, 'ices'), ('m', pl_sb_U_um_a_bysize, -2, 'a'), ('s', pl_sb_U_us_i_bysize, -2, 'i'), ('n', pl_sb_U_on_a_bysize, -2, 'a'), ('a', pl_sb_U_a_ae_bysize, None, 'e')):
            if word.lowered[-1] == lastlet:
                for k, v in d.items():
                    if word.lowered[-k:] in v:
                        return word[:numend] + post
        if self.classical_dict['ancient']:
            if word.lowered[-4:] == 'trix':
                return f'{word[:-1]}ces'
            if word.lowered[-3:] in ('eau', 'ieu'):
                return f'{word}x'
            if word.lowered[-3:] in ('ynx', 'inx', 'anx') and len(word) > 4:
                return f'{word[:-1]}ges'
            for lastlet, d, numend, post in (('n', pl_sb_C_en_ina_bysize, -2, 'ina'), ('x', pl_sb_C_ex_ices_bysize, -2, 'ices'), ('x', pl_sb_C_ix_ices_bysize, -2, 'ices'), ('m', pl_sb_C_um_a_bysize, -2, 'a'), ('s', pl_sb_C_us_i_bysize, -2, 'i'), ('s', pl_sb_C_us_us_bysize, None, ''), ('a', pl_sb_C_a_ae_bysize, None, 'e'), ('a', pl_sb_C_a_ata_bysize, None, 'ta'), ('s', pl_sb_C_is_ides_bysize, -1, 'des'), ('o', pl_sb_C_o_i_bysize, -1, 'i'), ('n', pl_sb_C_on_a_bysize, -2, 'a')):
                if word.lowered[-1] == lastlet:
                    for k, v in d.items():
                        if word.lowered[-k:] in v:
                            return word[:numend] + post
            for d, numend, post in ((pl_sb_C_i_bysize, None, 'i'), (pl_sb_C_im_bysize, None, 'im')):
                for k, v in d.items():
                    if word.lowered[-k:] in v:
                        return word[:numend] + post
        if lowered_last in pl_sb_singular_s_complete:
            return f'{word}es'
        for k, v in pl_sb_singular_s_bysize.items():
            if word.lowered[-k:] in v:
                return f'{word}es'
        if word.lowered[-2:] == 'es' and word[0] == word[0].upper():
            return f'{word}es'
        if word.lowered[-1] == 'z':
            for k, v in pl_sb_z_zes_bysize.items():
                if word.lowered[-k:] in v:
                    return f'{word}es'
            if word.lowered[-2:-1] != 'z':
                return f'{word}zes'
        if word.lowered[-2:] == 'ze':
            for k, v in pl_sb_ze_zes_bysize.items():
                if word.lowered[-k:] in v:
                    return f'{word}s'
        if word.lowered[-2:] in ('ch', 'sh', 'zz', 'ss') or word.lowered[-1] == 'x':
            return f'{word}es'
        if word.lowered[-3:] in ('elf', 'alf', 'olf'):
            return f'{word[:-1]}ves'
        if word.lowered[-3:] == 'eaf' and word.lowered[-4:-3] != 'd':
            return f'{word[:-1]}ves'
        if word.lowered[-4:] in ('nife', 'life', 'wife'):
            return f'{word[:-2]}ves'
        if word.lowered[-3:] == 'arf':
            return f'{word[:-1]}ves'
        if word.lowered[-1] == 'y':
            if word.lowered[-2:-1] in 'aeiou' or len(word) == 1:
                return f'{word}s'
            if self.classical_dict['names']:
                if word.lowered[-1] == 'y' and word[0] == word[0].upper():
                    return f'{word}s'
            return f'{word[:-1]}ies'
        if lowered_last in pl_sb_U_o_os_complete:
            return f'{word}s'
        for k, v in pl_sb_U_o_os_bysize.items():
            if word.lowered[-k:] in v:
                return f'{word}s'
        if word.lowered[-2:] in ('ao', 'eo', 'io', 'oo', 'uo'):
            return f'{word}s'
        if word.lowered[-1] == 'o':
            return f'{word}es'
        return f'{word}s'

    @classmethod
    def _handle_prepositional_phrase(cls, phrase, transform, sep):
        """
        Given a word or phrase possibly separated by sep, parse out
        the prepositional phrase and apply the transform to the word
        preceding the prepositional phrase.

        Raise ValueError if the pivot is not found or if at least two
        separators are not found.

        >>> engine._handle_prepositional_phrase("man-of-war", str.upper, '-')
        'MAN-of-war'
        >>> engine._handle_prepositional_phrase("man of war", str.upper, ' ')
        'MAN of war'
        """
        parts = phrase.split(sep)
        if len(parts) < 3:
            raise ValueError('Cannot handle words with fewer than two separators')
        pivot = cls._find_pivot(parts, pl_prep_list_da)
        transformed = transform(parts[pivot - 1]) or parts[pivot - 1]
        return ' '.join(parts[:pivot - 1] + [sep.join([transformed, parts[pivot], ''])]) + ' '.join(parts[pivot + 1:])

    def _handle_long_compounds(self, word: Words, count: int) -> Union[str, None]:
        """
        Handles the plural and singular for compound `Words` that
        have three or more words, based on the given count.

        >>> engine()._handle_long_compounds(Words("pair of scissors"), 2)
        'pairs of scissors'
        >>> engine()._handle_long_compounds(Words("men beyond hills"), 1)
        'man beyond hills'
        """
        inflection = self._sinoun if count == 1 else self._plnoun
        solutions = (' '.join(itertools.chain(leader, [inflection(cand, count), prep], trailer)) for leader, (cand, prep), trailer in windowed_complete(word.split_, 2) if prep in pl_prep_list_da)
        return next(solutions, None)

    @staticmethod
    def _find_pivot(words, candidates):
        pivots = (index for index in range(1, len(words) - 1) if words[index] in candidates)
        try:
            return next(pivots)
        except StopIteration:
            raise ValueError('No pivot found') from None

    def _pl_special_verb(self, word: str, count: Optional[Union[str, int]]=None) -> Union[str, bool]:
        if self.classical_dict['zero'] and str(count).lower() in pl_count_zero:
            return False
        count = self.get_count(count)
        if count == 1:
            return word
        value = self.ud_match(word, self.pl_v_user_defined)
        if value is not None:
            return value
        try:
            words = Words(word)
        except IndexError:
            return False
        if words.first in plverb_irregular_pres:
            return f'{plverb_irregular_pres[words.first]}{words[len(words.first):]}'
        if words.first in plverb_irregular_non_pres:
            return word
        if words.first.endswith("n't") and words.first[:-3] in plverb_irregular_pres:
            return f"{plverb_irregular_pres[words.first[:-3]]}n't{words[len(words.first):]}"
        if words.first.endswith("n't"):
            return word
        mo = PLVERB_SPECIAL_S_RE.search(word)
        if mo:
            return False
        if WHITESPACE.search(word):
            return False
        if words.lowered == 'quizzes':
            return 'quiz'
        if words.lowered[-4:] in ('ches', 'shes', 'zzes', 'sses') or words.lowered[-3:] == 'xes':
            return words[:-2]
        if words.lowered[-3:] == 'ies' and len(words) > 3:
            return words.lowered[:-3] + 'y'
        if words.last.lower() in pl_v_oes_oe or words.lowered[-4:] in pl_v_oes_oe_endings_size4 or words.lowered[-5:] in pl_v_oes_oe_endings_size5:
            return words[:-1]
        if words.lowered.endswith('oes') and len(words) > 3:
            return words.lowered[:-2]
        mo = ENDS_WITH_S.search(words)
        if mo:
            return mo.group(1)
        return False

    def _pl_general_verb(self, word: str, count: Optional[Union[str, int]]=None) -> str:
        count = self.get_count(count)
        if count == 1:
            return word
        mo = plverb_ambiguous_pres_keys.search(word)
        if mo:
            return f'{plverb_ambiguous_pres[mo.group(1).lower()]}{mo.group(2)}'
        mo = plverb_ambiguous_non_pres.search(word)
        if mo:
            return word
        return word

    def _pl_special_adjective(self, word: str, count: Optional[Union[str, int]]=None) -> Union[str, bool]:
        count = self.get_count(count)
        if count == 1:
            return word
        value = self.ud_match(word, self.pl_adj_user_defined)
        if value is not None:
            return value
        mo = pl_adj_special_keys.search(word)
        if mo:
            return pl_adj_special[mo.group(1).lower()]
        mo = pl_adj_poss_keys.search(word)
        if mo:
            return pl_adj_poss[mo.group(1).lower()]
        mo = ENDS_WITH_APOSTROPHE_S.search(word)
        if mo:
            pl = self.plural_noun(mo.group(1))
            trailing_s = '' if pl[-1] == 's' else 's'
            return f"{pl}'{trailing_s}"
        return False

    def _sinoun(self, word: str, count: Optional[Union[str, int]]=None, gender: Optional[str]=None) -> Union[str, bool]:
        count = self.get_count(count)
        if count == 2:
            return word
        try:
            if gender is None:
                gender = self.thegender
            elif gender not in singular_pronoun_genders:
                raise BadGenderError
        except (TypeError, IndexError) as err:
            raise BadGenderError from err
        value = self.ud_match(word, self.si_sb_user_defined)
        if value is not None:
            return value
        if word == '':
            return word
        if word in si_sb_ois_oi_case:
            return word[:-1]
        words = Words(word)
        if words.last.lower() in pl_sb_uninflected_complete:
            if len(words.split_) >= 3:
                return self._handle_long_compounds(words, count=1) or word
            return word
        if word in pl_sb_uninflected_caps:
            return word
        for k, v in pl_sb_uninflected_bysize.items():
            if words.lowered[-k:] in v:
                return word
        if self.classical_dict['herd'] and words.last.lower() in pl_sb_uninflected_herd:
            return word
        if words.last.lower() in pl_sb_C_us_us:
            return word if self.classical_dict['ancient'] else False
        mo = PL_SB_POSTFIX_ADJ_STEMS_RE.search(word)
        if mo and mo.group(2) != '':
            return f'{self._sinoun(mo.group(1), 1, gender=gender)}{mo.group(2)}'
        with contextlib.suppress(ValueError):
            return self._handle_prepositional_phrase(words.lowered, functools.partial(self._sinoun, count=1, gender=gender), ' ')
        with contextlib.suppress(ValueError):
            return self._handle_prepositional_phrase(words.lowered, functools.partial(self._sinoun, count=1, gender=gender), '-')
        for k, v in si_pron_acc_keys_bysize.items():
            if words.lowered[-k:] in v:
                for pk, pv in pl_prep_bysize.items():
                    if words.lowered[:pk] in pv:
                        if words.lowered.split() == [words.lowered[:pk], words.lowered[-k:]]:
                            return words.lowered[:-k] + get_si_pron('acc', words.lowered[-k:], gender)
        try:
            return get_si_pron('nom', words.lowered, gender)
        except KeyError:
            pass
        try:
            return get_si_pron('acc', words.lowered, gender)
        except KeyError:
            pass
        if words.last in si_sb_irregular_caps:
            llen = len(words.last)
            return f'{word[:-llen]}{si_sb_irregular_caps[words.last]}'
        if words.last.lower() in si_sb_irregular:
            llen = len(words.last.lower())
            return f'{word[:-llen]}{si_sb_irregular[words.last.lower()]}'
        dash_split = words.lowered.split('-')
        if ' '.join(dash_split[-2:]).lower() in si_sb_irregular_compound:
            llen = len(' '.join(dash_split[-2:]))
            return '{}{}'.format(word[:-llen], si_sb_irregular_compound[' '.join(dash_split[-2:]).lower()])
        if words.lowered[-5:] == 'quies':
            return word[:-3] + 'y'
        if words.lowered[-7:] == 'persons':
            return word[:-1]
        if words.lowered[-6:] == 'people':
            return word[:-4] + 'rson'
        if words.lowered[-4:] == 'mans':
            for k, v in si_sb_U_man_mans_bysize.items():
                if words.lowered[-k:] in v:
                    return word[:-1]
            for k, v in si_sb_U_man_mans_caps_bysize.items():
                if word[-k:] in v:
                    return word[:-1]
        if words.lowered[-3:] == 'men':
            return word[:-3] + 'man'
        if words.lowered[-4:] == 'mice':
            return word[:-4] + 'mouse'
        if words.lowered[-4:] == 'lice':
            v = si_sb_U_louse_lice_bysize.get(len(word))
            if v and words.lowered in v:
                return word[:-4] + 'louse'
        if words.lowered[-5:] == 'geese':
            return word[:-5] + 'goose'
        if words.lowered[-5:] == 'teeth':
            return word[:-5] + 'tooth'
        if words.lowered[-4:] == 'feet':
            return word[:-4] + 'foot'
        if words.lowered == 'dice':
            return 'die'
        if words.lowered[-4:] == 'ceps':
            return word
        if words.lowered[-3:] == 'zoa':
            return word[:-1] + 'on'
        for lastlet, d, unass_numend, post in (('s', si_sb_U_ch_chs_bysize, -1, ''), ('s', si_sb_U_ex_ices_bysize, -4, 'ex'), ('s', si_sb_U_ix_ices_bysize, -4, 'ix'), ('a', si_sb_U_um_a_bysize, -1, 'um'), ('i', si_sb_U_us_i_bysize, -1, 'us'), ('a', si_sb_U_on_a_bysize, -1, 'on'), ('e', si_sb_U_a_ae_bysize, -1, '')):
            if words.lowered[-1] == lastlet:
                for k, v in d.items():
                    if words.lowered[-k:] in v:
                        return word[:unass_numend] + post
        if self.classical_dict['ancient']:
            if words.lowered[-6:] == 'trices':
                return word[:-3] + 'x'
            if words.lowered[-4:] in ('eaux', 'ieux'):
                return word[:-1]
            if words.lowered[-5:] in ('ynges', 'inges', 'anges') and len(word) > 6:
                return word[:-3] + 'x'
            for lastlet, d, class_numend, post in (('a', si_sb_C_en_ina_bysize, -3, 'en'), ('s', si_sb_C_ex_ices_bysize, -4, 'ex'), ('s', si_sb_C_ix_ices_bysize, -4, 'ix'), ('a', si_sb_C_um_a_bysize, -1, 'um'), ('i', si_sb_C_us_i_bysize, -1, 'us'), ('s', pl_sb_C_us_us_bysize, None, ''), ('e', si_sb_C_a_ae_bysize, -1, ''), ('a', si_sb_C_a_ata_bysize, -2, ''), ('s', si_sb_C_is_ides_bysize, -3, 's'), ('i', si_sb_C_o_i_bysize, -1, 'o'), ('a', si_sb_C_on_a_bysize, -1, 'on'), ('m', si_sb_C_im_bysize, -2, ''), ('i', si_sb_C_i_bysize, -1, '')):
                if words.lowered[-1] == lastlet:
                    for k, v in d.items():
                        if words.lowered[-k:] in v:
                            return word[:class_numend] + post
        if words.lowered[-6:] == 'houses' or word in si_sb_uses_use_case or words.last.lower() in si_sb_uses_use:
            return word[:-1]
        if word in si_sb_ies_ie_case or words.last.lower() in si_sb_ies_ie:
            return word[:-1]
        if words.lowered[-5:] == 'shoes' or word in si_sb_oes_oe_case or words.last.lower() in si_sb_oes_oe:
            return word[:-1]
        if word in si_sb_sses_sse_case or words.last.lower() in si_sb_sses_sse:
            return word[:-1]
        if words.last.lower() in si_sb_singular_s_complete:
            return word[:-2]
        for k, v in si_sb_singular_s_bysize.items():
            if words.lowered[-k:] in v:
                return word[:-2]
        if words.lowered[-4:] == 'eses' and word[0] == word[0].upper():
            return word[:-2]
        if words.last.lower() in si_sb_z_zes:
            return word[:-2]
        if words.last.lower() in si_sb_zzes_zz:
            return word[:-2]
        if words.lowered[-4:] == 'zzes':
            return word[:-3]
        if word in si_sb_ches_che_case or words.last.lower() in si_sb_ches_che:
            return word[:-1]
        if words.lowered[-4:] in ('ches', 'shes'):
            return word[:-2]
        if words.last.lower() in si_sb_xes_xe:
            return word[:-1]
        if words.lowered[-3:] == 'xes':
            return word[:-2]
        if word in si_sb_ves_ve_case or words.last.lower() in si_sb_ves_ve:
            return word[:-1]
        if words.lowered[-3:] == 'ves':
            if words.lowered[-5:-3] in ('el', 'al', 'ol'):
                return word[:-3] + 'f'
            if words.lowered[-5:-3] == 'ea' and word[-6:-5] != 'd':
                return word[:-3] + 'f'
            if words.lowered[-5:-3] in ('ni', 'li', 'wi'):
                return word[:-3] + 'fe'
            if words.lowered[-5:-3] == 'ar':
                return word[:-3] + 'f'
        if words.lowered[-2:] == 'ys':
            if len(words.lowered) > 2 and words.lowered[-3] in 'aeiou':
                return word[:-1]
            if self.classical_dict['names']:
                if words.lowered[-2:] == 'ys' and word[0] == word[0].upper():
                    return word[:-1]
        if words.lowered[-3:] == 'ies':
            return word[:-3] + 'y'
        if words.lowered[-2:] == 'os':
            if words.last.lower() in si_sb_U_o_os_complete:
                return word[:-1]
            for k, v in si_sb_U_o_os_bysize.items():
                if words.lowered[-k:] in v:
                    return word[:-1]
            if words.lowered[-3:] in ('aos', 'eos', 'ios', 'oos', 'uos'):
                return word[:-1]
        if words.lowered[-3:] == 'oes':
            return word[:-2]
        if word in si_sb_es_is:
            return word[:-2] + 'is'
        if words.lowered[-1] == 's':
            return word[:-1]
        return False

    @typechecked
    def a(self, text: Word, count: Optional[Union[int, str, Any]]=1) -> str:
        """
        Return the appropriate indefinite article followed by text.

        The indefinite article is either 'a' or 'an'.

        If count is not one, then return count followed by text
        instead of 'a' or 'an'.

        Whitespace at the start and end is preserved.

        """
        mo = INDEFINITE_ARTICLE_TEST.search(text)
        if mo:
            word = mo.group(2)
            if not word:
                return text
            pre = mo.group(1)
            post = mo.group(3)
            result = self._indef_article(word, count)
            return f'{pre}{result}{post}'
        return ''
    an = a
    _indef_article_cases = ((A_ordinal_a, 'a'), (A_ordinal_an, 'an'), (A_explicit_an, 'an'), (SPECIAL_AN, 'an'), (SPECIAL_A, 'a'), (A_abbrev, 'an'), (SPECIAL_ABBREV_AN, 'an'), (SPECIAL_ABBREV_A, 'a'), (CONSONANTS, 'a'), (ARTICLE_SPECIAL_EU, 'a'), (ARTICLE_SPECIAL_ONCE, 'a'), (ARTICLE_SPECIAL_ONETIME, 'a'), (ARTICLE_SPECIAL_UNIT, 'a'), (ARTICLE_SPECIAL_UBA, 'a'), (ARTICLE_SPECIAL_UKR, 'a'), (A_explicit_a, 'a'), (SPECIAL_CAPITALS, 'a'), (VOWELS, 'an'), (A_y_cons, 'an'))

    def _indef_article(self, word: str, count: Union[int, str, Any]) -> str:
        mycount = self.get_count(count)
        if mycount != 1:
            return f'{count} {word}'
        value = self.ud_match(word, self.A_a_user_defined)
        if value is not None:
            return f'{value} {word}'
        matches = (f'{article} {word}' for regexen, article in self._indef_article_cases if regexen.search(word))
        fallback = f'a {word}'
        return next(matches, fallback)

    @typechecked
    def no(self, text: Word, count: Optional[Union[int, str]]=None) -> str:
        """
        If count is 0, no, zero or nil, return 'no' followed by the plural
        of text.

        If count is one of:
            1, a, an, one, each, every, this, that
            return count followed by text.

        Otherwise return count follow by the plural of text.

        In the return value count is always followed by a space.

        Whitespace at the start and end is preserved.

        """
        if count is None and self.persistent_count is not None:
            count = self.persistent_count
        if count is None:
            count = 0
        mo = PARTITION_WORD.search(text)
        if mo:
            pre = mo.group(1)
            word = mo.group(2)
            post = mo.group(3)
        else:
            pre = ''
            word = ''
            post = ''
        if str(count).lower() in pl_count_zero:
            count = 'no'
        return f'{pre}{count} {self.plural(word, count)}{post}'

    @typechecked
    def present_participle(self, word: Word) -> str:
        """
        Return the present participle for word.

        word is the 3rd person singular verb.

        """
        plv = self.plural_verb(word, 2)
        ans = plv
        for regexen, repl in PRESENT_PARTICIPLE_REPLACEMENTS:
            ans, num = regexen.subn(repl, plv)
            if num:
                return f'{ans}ing'
        return f'{ans}ing'

    @typechecked
    def ordinal(self, num: Union[Number, Word]) -> str:
        """
        Return the ordinal of num.

        >>> ordinal = engine().ordinal
        >>> ordinal(1)
        '1st'
        >>> ordinal('one')
        'first'
        """
        if DIGIT.match(str(num)):
            if isinstance(num, (float, int)) and int(num) == num:
                n = int(num)
            elif '.' in str(num):
                try:
                    n = int(str(num)[-1])
                except ValueError:
                    n = int(str(num)[:-1])
            else:
                n = int(num)
            try:
                post = nth[n % 100]
            except KeyError:
                post = nth[n % 10]
            return f'{num}{post}'
        else:
            str_num: str = num
            mo = ordinal_suff.search(str_num)
            if mo:
                post = ordinal[mo.group(1)]
                rval = ordinal_suff.sub(post, str_num)
            else:
                rval = f'{str_num}th'
            return rval

    def millfn(self, ind: int=0) -> str:
        if ind > len(mill) - 1:
            raise NumOutOfRangeError
        return mill[ind]

    def unitfn(self, units: int, mindex: int=0) -> str:
        return f'{unit[units]}{self.millfn(mindex)}'

    def tenfn(self, tens, units, mindex=0) -> str:
        if tens != 1:
            tens_part = ten[tens]
            if tens and units:
                hyphen = '-'
            else:
                hyphen = ''
            unit_part = unit[units]
            mill_part = self.millfn(mindex)
            return f'{tens_part}{hyphen}{unit_part}{mill_part}'
        return f'{teen[units]}{mill[mindex]}'

    def hundfn(self, hundreds: int, tens: int, units: int, mindex: int) -> str:
        if hundreds:
            andword = f' {self._number_args['andword']} ' if tens or units else ''
            return f'{unit[hundreds]} hundred{andword}{self.tenfn(tens, units)}{self.millfn(mindex)}, '
        if tens or units:
            return f'{self.tenfn(tens, units)}{self.millfn(mindex)}, '
        return ''

    def group1sub(self, mo: Match) -> str:
        units = int(mo.group(1))
        if units == 1:
            return f' {self._number_args['one']}, '
        elif units:
            return f'{unit[units]}, '
        else:
            return f' {self._number_args['zero']}, '

    def group1bsub(self, mo: Match) -> str:
        units = int(mo.group(1))
        if units:
            return f'{unit[units]}, '
        else:
            return f' {self._number_args['zero']}, '

    def group2sub(self, mo: Match) -> str:
        tens = int(mo.group(1))
        units = int(mo.group(2))
        if tens:
            return f'{self.tenfn(tens, units)}, '
        if units:
            return f' {self._number_args['zero']} {unit[units]}, '
        return f' {self._number_args['zero']} {self._number_args['zero']}, '

    def group3sub(self, mo: Match) -> str:
        hundreds = int(mo.group(1))
        tens = int(mo.group(2))
        units = int(mo.group(3))
        if hundreds == 1:
            hunword = f' {self._number_args['one']}'
        elif hundreds:
            hunword = str(unit[hundreds])
        else:
            hunword = f' {self._number_args['zero']}'
        if tens:
            tenword = self.tenfn(tens, units)
        elif units:
            tenword = f' {self._number_args['zero']} {unit[units]}'
        else:
            tenword = f' {self._number_args['zero']} {self._number_args['zero']}'
        return f'{hunword} {tenword}, '

    def hundsub(self, mo: Match) -> str:
        ret = self.hundfn(int(mo.group(1)), int(mo.group(2)), int(mo.group(3)), self.mill_count)
        self.mill_count += 1
        return ret

    def tensub(self, mo: Match) -> str:
        return f'{self.tenfn(int(mo.group(1)), int(mo.group(2)), self.mill_count)}, '

    def unitsub(self, mo: Match) -> str:
        return f'{self.unitfn(int(mo.group(1)), self.mill_count)}, '

    def enword(self, num: str, group: int) -> str:
        if group == 1:
            num = DIGIT_GROUP.sub(self.group1sub, num)
        elif group == 2:
            num = TWO_DIGITS.sub(self.group2sub, num)
            num = DIGIT_GROUP.sub(self.group1bsub, num, 1)
        elif group == 3:
            num = THREE_DIGITS.sub(self.group3sub, num)
            num = TWO_DIGITS.sub(self.group2sub, num, 1)
            num = DIGIT_GROUP.sub(self.group1sub, num, 1)
        elif int(num) == 0:
            num = self._number_args['zero']
        elif int(num) == 1:
            num = self._number_args['one']
        else:
            num = num.lstrip().lstrip('0')
            self.mill_count = 0
            mo = THREE_DIGITS_WORD.search(num)
            while mo:
                num = THREE_DIGITS_WORD.sub(self.hundsub, num, 1)
                mo = THREE_DIGITS_WORD.search(num)
            num = TWO_DIGITS_WORD.sub(self.tensub, num, 1)
            num = ONE_DIGIT_WORD.sub(self.unitsub, num, 1)
        return num

    @typechecked
    def number_to_words(self, num: Union[Number, Word], wantlist: bool=False, group: int=0, comma: Union[Falsish, str]=',', andword: str='and', zero: str='zero', one: str='one', decimal: Union[Falsish, str]='point', threshold: Optional[int]=None) -> Union[str, List[str]]:
        """
        Return a number in words.

        group = 1, 2 or 3 to group numbers before turning into words
        comma: define comma

        andword:
            word for 'and'. Can be set to ''.
            e.g. "one hundred and one" vs "one hundred one"

        zero: word for '0'
        one: word for '1'
        decimal: word for decimal point
        threshold: numbers above threshold not turned into words

        parameters not remembered from last call. Departure from Perl version.
        """
        self._number_args = {'andword': andword, 'zero': zero, 'one': one}
        num = str(num)
        if threshold is not None and float(num) > threshold:
            spnum = num.split('.', 1)
            while comma:
                spnum[0], n = FOUR_DIGIT_COMMA.subn('\\1,\\2', spnum[0])
                if n == 0:
                    break
            try:
                return f'{spnum[0]}.{spnum[1]}'
            except IndexError:
                return str(spnum[0])
        if group < 0 or group > 3:
            raise BadChunkingOptionError
        nowhite = num.lstrip()
        if nowhite[0] == '+':
            sign = 'plus'
        elif nowhite[0] == '-':
            sign = 'minus'
        else:
            sign = ''
        if num in nth_suff:
            num = zero
        myord = num[-2:] in nth_suff
        if myord:
            num = num[:-2]
        finalpoint = False
        if decimal:
            if group != 0:
                chunks = num.split('.')
            else:
                chunks = num.split('.', 1)
            if chunks[-1] == '':
                chunks = chunks[:-1]
                finalpoint = True
        else:
            chunks = [num]
        first: Union[int, str, bool] = 1
        loopstart = 0
        if chunks[0] == '':
            first = 0
            if len(chunks) > 1:
                loopstart = 1
        for i in range(loopstart, len(chunks)):
            chunk = chunks[i]
            chunk = NON_DIGIT.sub('', chunk)
            if chunk == '':
                chunk = '0'
            if group == 0 and (first == 0 or first == ''):
                chunk = self.enword(chunk, 1)
            else:
                chunk = self.enword(chunk, group)
            if chunk[-2:] == ', ':
                chunk = chunk[:-2]
            chunk = WHITESPACES_COMMA.sub(',', chunk)
            if group == 0 and first:
                chunk = COMMA_WORD.sub(f' {andword} \\1', chunk)
            chunk = WHITESPACES.sub(' ', chunk)
            chunk = chunk.strip()
            if first:
                first = ''
            chunks[i] = chunk
        numchunks = []
        if first != 0:
            numchunks = chunks[0].split(f'{comma} ')
        if myord and numchunks:
            mo = ordinal_suff.search(numchunks[-1])
            if mo:
                numchunks[-1] = ordinal_suff.sub(ordinal[mo.group(1)], numchunks[-1])
            else:
                numchunks[-1] += 'th'
        for chunk in chunks[1:]:
            numchunks.append(decimal)
            numchunks.extend(chunk.split(f'{comma} '))
        if finalpoint:
            numchunks.append(decimal)
        if wantlist:
            if sign:
                numchunks = [sign] + numchunks
            return numchunks
        elif group:
            signout = f'{sign} ' if sign else ''
            return f'{signout}{', '.join(numchunks)}'
        else:
            signout = f'{sign} ' if sign else ''
            num = f'{signout}{numchunks.pop(0)}'
            if decimal is None:
                first = True
            else:
                first = not num.endswith(decimal)
            for nc in numchunks:
                if nc == decimal:
                    num += f' {nc}'
                    first = 0
                elif first:
                    num += f'{comma} {nc}'
                else:
                    num += f' {nc}'
            return num

    @typechecked
    def join(self, words: Optional[Sequence[Word]], sep: Optional[str]=None, sep_spaced: bool=True, final_sep: Optional[str]=None, conj: str='and', conj_spaced: bool=True) -> str:
        """
        Join words into a list.

        e.g. join(['ant', 'bee', 'fly']) returns 'ant, bee, and fly'

        options:
        conj: replacement for 'and'
        sep: separator. default ',', unless ',' is in the list then ';'
        final_sep: final separator. default ',', unless ',' is in the list then ';'
        conj_spaced: boolean. Should conj have spaces around it

        """
        if not words:
            return ''
        if len(words) == 1:
            return words[0]
        if conj_spaced:
            if conj == '':
                conj = ' '
            else:
                conj = f' {conj} '
        if len(words) == 2:
            return f'{words[0]}{conj}{words[1]}'
        if sep is None:
            if ',' in ''.join(words):
                sep = ';'
            else:
                sep = ','
        if final_sep is None:
            final_sep = sep
        final_sep = f'{final_sep}{conj}'
        if sep_spaced:
            sep += ' '
        return f'{sep.join(words[0:-1])}{final_sep}{words[-1]}'