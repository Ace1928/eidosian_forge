import argparse
import os, sys
import re
import string
import traceback
import itertools
from elftools import __version__
from elftools.common.exceptions import ELFError
from elftools.common.utils import bytes2str, iterbytes
from elftools.elf.elffile import ELFFile
from elftools.elf.dynamic import DynamicSection, DynamicSegment
from elftools.elf.enums import ENUM_D_TAG
from elftools.elf.segments import InterpSegment
from elftools.elf.sections import (
from elftools.elf.gnuversions import (
from elftools.elf.relocation import RelocationSection
from elftools.elf.descriptions import (
from elftools.elf.constants import E_FLAGS
from elftools.elf.constants import E_FLAGS_MASKS
from elftools.elf.constants import SH_FLAGS
from elftools.elf.constants import SHN_INDICES
from elftools.dwarf.dwarfinfo import DWARFInfo
from elftools.dwarf.descriptions import (
from elftools.dwarf.constants import (
from elftools.dwarf.locationlists import LocationParser, LocationEntry, LocationViewPair, BaseAddressEntry as LocBaseAddressEntry, LocationListsPair
from elftools.dwarf.ranges import RangeEntry, BaseAddressEntry as RangeBaseAddressEntry, RangeListsPair
from elftools.dwarf.callframe import CIE, FDE, ZERO
from elftools.ehabi.ehabiinfo import CorruptEHABIEntry, CannotUnwindEHABIEntry, GenericEHABIEntry
from elftools.dwarf.enums import ENUM_DW_UT
def _print_version_section_header(self, version_section, name, lead0x=True, indent=1):
    """ Print a section header of one version related section (versym,
            verneed or verdef) with some options to accomodate readelf
            little differences between each header (e.g. indentation
            and 0x prefixing).
        """
    if hasattr(version_section, 'num_versions'):
        num_entries = version_section.num_versions()
    else:
        num_entries = version_section.num_symbols()
    self._emitline("\n%s section '%s' contains %d %s:" % (name, version_section.name, num_entries, 'entry' if num_entries == 1 else 'entries'))
    self._emitline('%sAddr: %s  Offset: %s  Link: %i (%s)' % (' ' * indent, self._format_hex(version_section['sh_addr'], fieldsize=16, lead0x=lead0x), self._format_hex(version_section['sh_offset'], fieldsize=8, lead0x=True), version_section['sh_link'], self.elffile.get_section(version_section['sh_link']).name))