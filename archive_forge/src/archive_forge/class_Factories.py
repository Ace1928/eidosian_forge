import os
import re
from pathlib import Path
from typing import Mapping
import configparser
import pytest
from ase.calculators.calculator import (names as calculator_names,
class Factories:
    all_calculators = set(calculator_names)
    builtin_calculators = {'eam', 'emt', 'ff', 'lj', 'morse', 'tip3p', 'tip4p'}
    autoenabled_calculators = {'asap'} | builtin_calculators
    monkeypatch_calculator_constructors = {'ace', 'aims', 'amber', 'crystal', 'demon', 'demonnano', 'dftd3', 'dmol', 'exciting', 'fleur', 'gamess_us', 'gaussian', 'gulp', 'hotbit', 'lammpslib', 'mopac', 'onetep', 'orca', 'Psi4', 'qchem', 'turbomole'}

    def __init__(self, requested_calculators):
        executable_config_paths, executables = get_testing_executables()
        assert isinstance(executables, Mapping), executables
        self.executables = executables
        self.executable_config_paths = executable_config_paths
        datafiles_module = None
        datafiles = {}
        try:
            import asetest as datafiles_module
        except ImportError:
            pass
        else:
            datafiles.update(datafiles_module.datafiles.paths)
            datafiles_module = datafiles_module
        self.datafiles_module = datafiles_module
        self.datafiles = datafiles
        factories = {}
        for name, cls in factory_classes.items():
            try:
                factory = cls.fromconfig(self)
            except (NotInstalled, KeyError):
                pass
            else:
                factories[name] = factory
        self.factories = factories
        requested_calculators = set(requested_calculators)
        if 'auto' in requested_calculators:
            requested_calculators.remove('auto')
            requested_calculators |= set(self.factories)
        self.requested_calculators = requested_calculators
        for name in self.requested_calculators:
            if name not in self.all_calculators:
                raise NoSuchCalculator(name)

    def installed(self, name):
        return name in self.builtin_calculators | set(self.factories)

    def is_adhoc(self, name):
        return name not in factory_classes

    def optional(self, name):
        return name not in self.builtin_calculators

    def enabled(self, name):
        auto = name in self.autoenabled_calculators and self.installed(name)
        return auto or name in self.requested_calculators

    def require(self, name):
        assert name in calculator_names
        if not self.installed(name) and (not self.is_adhoc(name)):
            pytest.skip(f'Not installed: {name}')
        if name not in self.requested_calculators:
            pytest.skip(f'Use --calculators={name} to enable')

    def __getitem__(self, name):
        return self.factories[name]

    def monkeypatch_disabled_calculators(self):
        test_calculator_names = self.autoenabled_calculators | self.builtin_calculators | self.requested_calculators
        disable_names = self.monkeypatch_calculator_constructors - test_calculator_names
        for name in disable_names:
            try:
                cls = get_calculator_class(name)
            except ImportError:
                pass
            else:

                def get_mock_init(name):

                    def mock_init(obj, *args, **kwargs):
                        pytest.skip(f'use --calculators={name} to enable')
                    return mock_init

                def mock_del(obj):
                    pass
                cls.__init__ = get_mock_init(name)
                cls.__del__ = mock_del