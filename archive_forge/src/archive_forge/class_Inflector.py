from __future__ import absolute_import, division, print_function
import errno
import glob
import json
import os
import re
import re
import re
class Inflector(object):
    """
    Inflector - perform inflections
    """

    def __init__(self):
        self.inflections = Inflections()
        self.inflections.plural('$', 's')
        self.inflections.plural('(?i)([sxz]|[cs]h)$', '\\1es')
        self.inflections.plural('(?i)([^aeiouy]o)$', '\\1es')
        self.inflections.plural('(?i)([^aeiouy])y$', '\\1ies')
        self.inflections.singular('(?i)s$', '')
        self.inflections.singular('(?i)(ss)$', '\\1')
        self.inflections.singular('([sxz]|[cs]h)es$', '\\1')
        self.inflections.singular('([^aeiouy]o)es$', '\\1')
        self.inflections.singular('(?i)([^aeiouy])ies$', '\\1y')
        self.inflections.irregular('child', 'children')
        self.inflections.irregular('man', 'men')
        self.inflections.irregular('medium', 'media')
        self.inflections.irregular('move', 'moves')
        self.inflections.irregular('person', 'people')
        self.inflections.irregular('self', 'selves')
        self.inflections.irregular('sex', 'sexes')
        self.inflections.irregular('erratum', 'errata')
        self.inflections.uncountable('equipment', 'information', 'money', 'species', 'series', 'fish', 'sheep', 'police')

    def pluralize(self, word):
        """
        Pluralize a word.
        """
        return self._apply_inflections(word, self.inflections.plurals)

    def singularize(self, word):
        """
        Singularize a word.
        """
        return self._apply_inflections(word, self.inflections.singulars)

    def _apply_inflections(self, word, rules):
        result = word
        if word != '' and result.lower() not in self.inflections.uncountables:
            for rule, replacement in rules:
                result = re.sub(rule, replacement, result)
                if result != word:
                    break
        return result