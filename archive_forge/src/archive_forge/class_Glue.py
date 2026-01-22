import os
from itertools import chain
import nltk
from nltk.internals import Counter
from nltk.sem import drt, linearlogic
from nltk.sem.logic import (
from nltk.tag import BigramTagger, RegexpTagger, TrigramTagger, UnigramTagger
class Glue:

    def __init__(self, semtype_file=None, remove_duplicates=False, depparser=None, verbose=False):
        self.verbose = verbose
        self.remove_duplicates = remove_duplicates
        self.depparser = depparser
        from nltk import Prover9
        self.prover = Prover9()
        if semtype_file:
            self.semtype_file = semtype_file
        else:
            self.semtype_file = os.path.join('grammars', 'sample_grammars', 'glue.semtype')

    def train_depparser(self, depgraphs=None):
        if depgraphs:
            self.depparser.train(depgraphs)
        else:
            self.depparser.train_from_file(nltk.data.find(os.path.join('grammars', 'sample_grammars', 'glue_train.conll')))

    def parse_to_meaning(self, sentence):
        readings = []
        for agenda in self.parse_to_compiled(sentence):
            readings.extend(self.get_readings(agenda))
        return readings

    def get_readings(self, agenda):
        readings = []
        agenda_length = len(agenda)
        atomics = dict()
        nonatomics = dict()
        while agenda:
            cur = agenda.pop()
            glue_simp = cur.glue.simplify()
            if isinstance(glue_simp, linearlogic.ImpExpression):
                for key in atomics:
                    try:
                        if isinstance(cur.glue, linearlogic.ApplicationExpression):
                            bindings = cur.glue.bindings
                        else:
                            bindings = linearlogic.BindingDict()
                        glue_simp.antecedent.unify(key, bindings)
                        for atomic in atomics[key]:
                            if not cur.indices & atomic.indices:
                                try:
                                    agenda.append(cur.applyto(atomic))
                                except linearlogic.LinearLogicApplicationException:
                                    pass
                    except linearlogic.UnificationException:
                        pass
                try:
                    nonatomics[glue_simp.antecedent].append(cur)
                except KeyError:
                    nonatomics[glue_simp.antecedent] = [cur]
            else:
                for key in nonatomics:
                    for nonatomic in nonatomics[key]:
                        try:
                            if isinstance(nonatomic.glue, linearlogic.ApplicationExpression):
                                bindings = nonatomic.glue.bindings
                            else:
                                bindings = linearlogic.BindingDict()
                            glue_simp.unify(key, bindings)
                            if not cur.indices & nonatomic.indices:
                                try:
                                    agenda.append(nonatomic.applyto(cur))
                                except linearlogic.LinearLogicApplicationException:
                                    pass
                        except linearlogic.UnificationException:
                            pass
                try:
                    atomics[glue_simp].append(cur)
                except KeyError:
                    atomics[glue_simp] = [cur]
        for entry in atomics:
            for gf in atomics[entry]:
                if len(gf.indices) == agenda_length:
                    self._add_to_reading_list(gf, readings)
        for entry in nonatomics:
            for gf in nonatomics[entry]:
                if len(gf.indices) == agenda_length:
                    self._add_to_reading_list(gf, readings)
        return readings

    def _add_to_reading_list(self, glueformula, reading_list):
        add_reading = True
        if self.remove_duplicates:
            for reading in reading_list:
                try:
                    if reading.equiv(glueformula.meaning, self.prover):
                        add_reading = False
                        break
                except Exception as e:
                    print('Error when checking logical equality of statements', e)
        if add_reading:
            reading_list.append(glueformula.meaning)

    def parse_to_compiled(self, sentence):
        gfls = [self.depgraph_to_glue(dg) for dg in self.dep_parse(sentence)]
        return [self.gfl_to_compiled(gfl) for gfl in gfls]

    def dep_parse(self, sentence):
        """
        Return a dependency graph for the sentence.

        :param sentence: the sentence to be parsed
        :type sentence: list(str)
        :rtype: DependencyGraph
        """
        if self.depparser is None:
            from nltk.parse import MaltParser
            self.depparser = MaltParser(tagger=self.get_pos_tagger())
        if not self.depparser._trained:
            self.train_depparser()
        return self.depparser.parse(sentence, verbose=self.verbose)

    def depgraph_to_glue(self, depgraph):
        return self.get_glue_dict().to_glueformula_list(depgraph)

    def get_glue_dict(self):
        return GlueDict(self.semtype_file)

    def gfl_to_compiled(self, gfl):
        index_counter = Counter()
        return_list = []
        for gf in gfl:
            return_list.extend(gf.compile(index_counter))
        if self.verbose:
            print('Compiled Glue Premises:')
            for cgf in return_list:
                print(cgf)
        return return_list

    def get_pos_tagger(self):
        from nltk.corpus import brown
        regexp_tagger = RegexpTagger([('^-?[0-9]+(\\.[0-9]+)?$', 'CD'), ('(The|the|A|a|An|an)$', 'AT'), ('.*able$', 'JJ'), ('.*ness$', 'NN'), ('.*ly$', 'RB'), ('.*s$', 'NNS'), ('.*ing$', 'VBG'), ('.*ed$', 'VBD'), ('.*', 'NN')])
        brown_train = brown.tagged_sents(categories='news')
        unigram_tagger = UnigramTagger(brown_train, backoff=regexp_tagger)
        bigram_tagger = BigramTagger(brown_train, backoff=unigram_tagger)
        trigram_tagger = TrigramTagger(brown_train, backoff=bigram_tagger)
        main_tagger = RegexpTagger([('(A|a|An|an)$', 'ex_quant'), ('(Every|every|All|all)$', 'univ_quant')], backoff=trigram_tagger)
        return main_tagger