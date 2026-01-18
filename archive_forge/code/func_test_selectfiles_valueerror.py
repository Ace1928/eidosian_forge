import os
import copy
import simplejson
import glob
import os.path as op
from subprocess import Popen
import hashlib
from collections import namedtuple
import pytest
import nipype
import nipype.interfaces.io as nio
from nipype.interfaces.base.traits_extension import isdefined
from nipype.interfaces.base import Undefined, TraitError
from nipype.utils.filemanip import dist_is_editable
from subprocess import check_call, CalledProcessError
def test_selectfiles_valueerror():
    """Test ValueError when force_lists has field that isn't in template."""
    base_dir = op.dirname(nipype.__file__)
    templates = {'model': 'interfaces/{package}/model.py', 'preprocess': 'interfaces/{package}/pre*.py'}
    force_lists = ['model', 'preprocess', 'registration']
    sf = nio.SelectFiles(templates, base_directory=base_dir, force_lists=force_lists)
    with pytest.raises(ValueError):
        sf.run()