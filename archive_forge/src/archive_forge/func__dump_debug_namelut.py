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
def _dump_debug_namelut(self, what):
    """
        Dump the debug pubnames section.
        """
    if what == 'pubnames':
        namelut = self._dwarfinfo.get_pubnames()
        section = self._dwarfinfo.debug_pubnames_sec
    else:
        namelut = self._dwarfinfo.get_pubtypes()
        section = self._dwarfinfo.debug_pubtypes_sec
    if namelut is None or len(namelut) == 0:
        return
    self._emitline('Contents of the %s section:' % section.name)
    self._emitline()
    cu_headers = namelut.get_cu_headers()
    for cu_hdr, (cu_ofs, items) in izip(cu_headers, itertools.groupby(namelut.items(), key=lambda x: x[1].cu_ofs)):
        self._emitline('  Length:                              %d' % cu_hdr.unit_length)
        self._emitline('  Version:                             %d' % cu_hdr.version)
        self._emitline('  Offset into .debug_info section:     0x%x' % cu_hdr.debug_info_offset)
        self._emitline('  Size of area in .debug_info section: %d' % cu_hdr.debug_info_length)
        self._emitline()
        self._emitline('    Offset  Name')
        for item in items:
            self._emitline('    %x          %s' % (item[1].die_ofs - cu_ofs, item[0]))
    self._emitline()