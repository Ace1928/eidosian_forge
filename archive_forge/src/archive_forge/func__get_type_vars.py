import abc
import collections
import collections.abc
import operator
import sys
import typing
def _get_type_vars(self, tvars):
    if self.__origin__ and self.__parameters__:
        typing._get_type_vars(self.__parameters__, tvars)