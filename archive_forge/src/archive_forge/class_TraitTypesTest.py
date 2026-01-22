import os
import sys
import tempfile
import textwrap
import shutil
import subprocess
import unittest
from traits.api import (
from traits.testing.optional_dependencies import requires_numpy
class TraitTypesTest(unittest.TestCase):

    def test_traits_shared_transient(self):

        class LazyProperty(TraitType):
            default_value_type = DefaultValue.constant

            def get(self, obj, name):
                return 1729
        self.assertFalse(Float().transient)
        LazyProperty().as_ctrait()
        self.assertFalse(Float().transient)

    @requires_numpy
    def test_numpy_validators_loaded_if_numpy_present(self):
        test_script = textwrap.dedent('\n            from traits.trait_types import float_fast_validate\n            import numpy\n\n            if numpy.floating in float_fast_validate:\n                print("Success")\n            else:\n                print("Failure")\n        ')
        this_python = sys.executable
        tmpdir = tempfile.mkdtemp()
        try:
            tmpfile = os.path.join(tmpdir, 'test_script.py')
            with open(tmpfile, 'w', encoding='utf-8') as f:
                f.write(test_script)
            cmd = [this_python, tmpfile]
            output = subprocess.check_output(cmd).decode('utf-8')
        finally:
            shutil.rmtree(tmpdir)
        self.assertEqual(output.strip(), 'Success')

    def test_default_value_in_init(self):

        class MyTraitType(TraitType):
            pass
        trait_type = MyTraitType(default_value=23)
        self.assertEqual(trait_type.get_default_value(), (DefaultValue.constant, 23))
        trait_type = MyTraitType(default_value=None)
        self.assertEqual(trait_type.get_default_value(), (DefaultValue.constant, None))
        trait_type = MyTraitType()
        self.assertEqual(trait_type.get_default_value(), (DefaultValue.constant, Undefined))
        trait_type = MyTraitType(default_value=NoDefaultSpecified)
        self.assertEqual(trait_type.get_default_value(), (DefaultValue.constant, Undefined))

    def test_disallowed_default_value(self):

        class MyTraitType(TraitType):
            default_value_type = DefaultValue.disallow
        trait_type = MyTraitType()
        self.assertEqual(trait_type.get_default_value(), (DefaultValue.disallow, Undefined))
        ctrait = trait_type.as_ctrait()
        self.assertEqual(ctrait.default_value(), (DefaultValue.disallow, Undefined))
        self.assertEqual(ctrait.default_kind, 'invalid')
        self.assertEqual(ctrait.default, Undefined)
        with self.assertRaises(ValueError):
            ctrait.default_value_for(None, '<dummy>')