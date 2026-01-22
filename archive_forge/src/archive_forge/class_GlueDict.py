import os
from itertools import chain
import nltk
from nltk.internals import Counter
from nltk.sem import drt, linearlogic
from nltk.sem.logic import (
from nltk.tag import BigramTagger, RegexpTagger, TrigramTagger, UnigramTagger
class GlueDict(dict):

    def __init__(self, filename, encoding=None):
        self.filename = filename
        self.file_encoding = encoding
        self.read_file()

    def read_file(self, empty_first=True):
        if empty_first:
            self.clear()
        try:
            contents = nltk.data.load(self.filename, format='text', encoding=self.file_encoding)
        except LookupError as e:
            try:
                contents = nltk.data.load('file:' + self.filename, format='text', encoding=self.file_encoding)
            except LookupError:
                raise e
        lines = contents.splitlines()
        for line in lines:
            line = line.strip()
            if not len(line):
                continue
            if line[0] == '#':
                continue
            parts = line.split(' : ', 2)
            glue_formulas = []
            paren_count = 0
            tuple_start = 0
            tuple_comma = 0
            relationships = None
            if len(parts) > 1:
                for i, c in enumerate(parts[1]):
                    if c == '(':
                        if paren_count == 0:
                            tuple_start = i + 1
                        paren_count += 1
                    elif c == ')':
                        paren_count -= 1
                        if paren_count == 0:
                            meaning_term = parts[1][tuple_start:tuple_comma]
                            glue_term = parts[1][tuple_comma + 1:i]
                            glue_formulas.append([meaning_term, glue_term])
                    elif c == ',':
                        if paren_count == 1:
                            tuple_comma = i
                    elif c == '#':
                        if paren_count != 0:
                            raise RuntimeError('Formula syntax is incorrect for entry ' + line)
                        break
            if len(parts) > 2:
                rel_start = parts[2].index('[') + 1
                rel_end = parts[2].index(']')
                if rel_start == rel_end:
                    relationships = frozenset()
                else:
                    relationships = frozenset((r.strip() for r in parts[2][rel_start:rel_end].split(',')))
            try:
                start_inheritance = parts[0].index('(')
                end_inheritance = parts[0].index(')')
                sem = parts[0][:start_inheritance].strip()
                supertype = parts[0][start_inheritance + 1:end_inheritance]
            except:
                sem = parts[0].strip()
                supertype = None
            if sem not in self:
                self[sem] = {}
            if relationships is None:
                if supertype:
                    for rels in self[supertype]:
                        if rels not in self[sem]:
                            self[sem][rels] = []
                        glue = self[supertype][rels]
                        self[sem][rels].extend(glue)
                        self[sem][rels].extend(glue_formulas)
                else:
                    if None not in self[sem]:
                        self[sem][None] = []
                    self[sem][None].extend(glue_formulas)
            else:
                if relationships not in self[sem]:
                    self[sem][relationships] = []
                if supertype:
                    self[sem][relationships].extend(self[supertype][relationships])
                self[sem][relationships].extend(glue_formulas)

    def __str__(self):
        accum = ''
        for pos in self:
            str_pos = '%s' % pos
            for relset in self[pos]:
                i = 1
                for gf in self[pos][relset]:
                    if i == 1:
                        accum += str_pos + ': '
                    else:
                        accum += ' ' * (len(str_pos) + 2)
                    accum += '%s' % gf
                    if relset and i == len(self[pos][relset]):
                        accum += ' : %s' % relset
                    accum += '\n'
                    i += 1
        return accum

    def to_glueformula_list(self, depgraph, node=None, counter=None, verbose=False):
        if node is None:
            top = depgraph.nodes[0]
            depList = list(chain.from_iterable(top['deps'].values()))
            root = depgraph.nodes[depList[0]]
            return self.to_glueformula_list(depgraph, root, Counter(), verbose)
        glueformulas = self.lookup(node, depgraph, counter)
        for dep_idx in chain.from_iterable(node['deps'].values()):
            dep = depgraph.nodes[dep_idx]
            glueformulas.extend(self.to_glueformula_list(depgraph, dep, counter, verbose))
        return glueformulas

    def lookup(self, node, depgraph, counter):
        semtype_names = self.get_semtypes(node)
        semtype = None
        for name in semtype_names:
            if name in self:
                semtype = self[name]
                break
        if semtype is None:
            return []
        self.add_missing_dependencies(node, depgraph)
        lookup = self._lookup_semtype_option(semtype, node, depgraph)
        if not len(lookup):
            raise KeyError("There is no GlueDict entry for sem type of '%s' with tag '%s', and rel '%s'" % (node['word'], node['tag'], node['rel']))
        return self.get_glueformulas_from_semtype_entry(lookup, node['word'], node, depgraph, counter)

    def add_missing_dependencies(self, node, depgraph):
        rel = node['rel'].lower()
        if rel == 'main':
            headnode = depgraph.nodes[node['head']]
            subj = self.lookup_unique('subj', headnode, depgraph)
            relation = subj['rel']
            node['deps'].setdefault(relation, [])
            node['deps'][relation].append(subj['address'])

    def _lookup_semtype_option(self, semtype, node, depgraph):
        relationships = frozenset((depgraph.nodes[dep]['rel'].lower() for dep in chain.from_iterable(node['deps'].values()) if depgraph.nodes[dep]['rel'].lower() not in OPTIONAL_RELATIONSHIPS))
        try:
            lookup = semtype[relationships]
        except KeyError:
            best_match = frozenset()
            for relset_option in set(semtype) - {None}:
                if len(relset_option) > len(best_match) and relset_option < relationships:
                    best_match = relset_option
            if not best_match:
                if None in semtype:
                    best_match = None
                else:
                    return None
            lookup = semtype[best_match]
        return lookup

    def get_semtypes(self, node):
        """
        Based on the node, return a list of plausible semtypes in order of
        plausibility.
        """
        rel = node['rel'].lower()
        word = node['word'].lower()
        if rel == 'spec':
            if word in SPEC_SEMTYPES:
                return [SPEC_SEMTYPES[word]]
            else:
                return [SPEC_SEMTYPES['default']]
        elif rel in ['nmod', 'vmod']:
            return [node['tag'], rel]
        else:
            return [node['tag']]

    def get_glueformulas_from_semtype_entry(self, lookup, word, node, depgraph, counter):
        glueformulas = []
        glueFormulaFactory = self.get_GlueFormula_factory()
        for meaning, glue in lookup:
            gf = glueFormulaFactory(self.get_meaning_formula(meaning, word), glue)
            if not len(glueformulas):
                gf.word = word
            else:
                gf.word = f'{word}{len(glueformulas) + 1}'
            gf.glue = self.initialize_labels(gf.glue, node, depgraph, counter.get())
            glueformulas.append(gf)
        return glueformulas

    def get_meaning_formula(self, generic, word):
        """
        :param generic: A meaning formula string containing the
            parameter "<word>"
        :param word: The actual word to be replace "<word>"
        """
        word = word.replace('.', '')
        return generic.replace('<word>', word)

    def initialize_labels(self, expr, node, depgraph, unique_index):
        if isinstance(expr, linearlogic.AtomicExpression):
            name = self.find_label_name(expr.name.lower(), node, depgraph, unique_index)
            if name[0].isupper():
                return linearlogic.VariableExpression(name)
            else:
                return linearlogic.ConstantExpression(name)
        else:
            return linearlogic.ImpExpression(self.initialize_labels(expr.antecedent, node, depgraph, unique_index), self.initialize_labels(expr.consequent, node, depgraph, unique_index))

    def find_label_name(self, name, node, depgraph, unique_index):
        try:
            dot = name.index('.')
            before_dot = name[:dot]
            after_dot = name[dot + 1:]
            if before_dot == 'super':
                return self.find_label_name(after_dot, depgraph.nodes[node['head']], depgraph, unique_index)
            else:
                return self.find_label_name(after_dot, self.lookup_unique(before_dot, node, depgraph), depgraph, unique_index)
        except ValueError:
            lbl = self.get_label(node)
            if name == 'f':
                return lbl
            elif name == 'v':
                return '%sv' % lbl
            elif name == 'r':
                return '%sr' % lbl
            elif name == 'super':
                return self.get_label(depgraph.nodes[node['head']])
            elif name == 'var':
                return f'{lbl.upper()}{unique_index}'
            elif name == 'a':
                return self.get_label(self.lookup_unique('conja', node, depgraph))
            elif name == 'b':
                return self.get_label(self.lookup_unique('conjb', node, depgraph))
            else:
                return self.get_label(self.lookup_unique(name, node, depgraph))

    def get_label(self, node):
        """
        Pick an alphabetic character as identifier for an entity in the model.

        :param value: where to index into the list of characters
        :type value: int
        """
        value = node['address']
        letter = ['f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'a', 'b', 'c', 'd', 'e'][value - 1]
        num = int(value) // 26
        if num > 0:
            return letter + str(num)
        else:
            return letter

    def lookup_unique(self, rel, node, depgraph):
        """
        Lookup 'key'. There should be exactly one item in the associated relation.
        """
        deps = [depgraph.nodes[dep] for dep in chain.from_iterable(node['deps'].values()) if depgraph.nodes[dep]['rel'].lower() == rel.lower()]
        if len(deps) == 0:
            raise KeyError("'{}' doesn't contain a feature '{}'".format(node['word'], rel))
        elif len(deps) > 1:
            raise KeyError("'{}' should only have one feature '{}'".format(node['word'], rel))
        else:
            return deps[0]

    def get_GlueFormula_factory(self):
        return GlueFormula