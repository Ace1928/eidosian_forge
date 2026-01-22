import os
import sys
from os.path import abspath, dirname, normpath, join
from pyomo.common.fileutils import import_file
from pyomo.repn.tests.lp_diff import load_and_compare_lp_baseline
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import yaml, yaml_available, yaml_load_args
import pyomo.opt
from pyomo.environ import SolverFactory, TransformationFactory
class Labeler(type):

    def __new__(meta, name, bases, attrs):
        for key in attrs.keys():
            if key.startswith('test_'):
                for base in bases:
                    original = getattr(base, key, None)
                    if original is not None:
                        copy = copyfunc(original)
                        copy.__doc__ = attrs[key].__doc__ + ' (%s)' % copy.__name__
                        attrs[key] = copy
                        break
        for base in bases:
            for key in dir(base):
                if key.startswith('test_') and key not in attrs:
                    original = getattr(base, key)
                    copy = copyfunc(original)
                    copy.__doc__ = original.__doc__ + ' (%s)' % name
                    attrs[key] = copy
        return type.__new__(meta, name, bases, attrs)