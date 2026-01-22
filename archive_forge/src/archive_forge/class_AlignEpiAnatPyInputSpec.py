import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class AlignEpiAnatPyInputSpec(AFNIPythonCommandInputSpec):
    in_file = File(desc='EPI dataset to align', argstr='-epi %s', mandatory=True, exists=True, copyfile=False)
    anat = File(desc='name of structural dataset', argstr='-anat %s', mandatory=True, exists=True, copyfile=False)
    epi_base = traits.Either(traits.Range(low=0), traits.Enum('mean', 'median', 'max'), desc='the epi base used in alignmentshould be one of (0/mean/median/max/subbrick#)', mandatory=True, argstr='-epi_base %s')
    anat2epi = traits.Bool(desc='align anatomical to EPI dataset (default)', argstr='-anat2epi')
    epi2anat = traits.Bool(desc='align EPI to anatomical dataset', argstr='-epi2anat')
    save_skullstrip = traits.Bool(desc='save skull-stripped (not aligned)', argstr='-save_skullstrip')
    suffix = traits.Str('_al', desc='append suffix to the original anat/epi dataset to usein the resulting dataset names (default is "_al")', usedefault=True, argstr='-suffix %s')
    epi_strip = traits.Enum(('3dSkullStrip', '3dAutomask', 'None'), desc='method to mask brain in EPI datashould be one of[3dSkullStrip]/3dAutomask/None)', argstr='-epi_strip %s')
    volreg = traits.Enum('on', 'off', usedefault=True, desc="do volume registration on EPI dataset before alignmentshould be 'on' or 'off', defaults to 'on'", argstr='-volreg %s')
    tshift = traits.Enum('on', 'off', usedefault=True, desc="do time shifting of EPI dataset before alignmentshould be 'on' or 'off', defaults to 'on'", argstr='-tshift %s')