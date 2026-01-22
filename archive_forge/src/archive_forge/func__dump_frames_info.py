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
def _dump_frames_info(self, section, cfi_entries):
    """ Dump the raw call frame info in a section.

            `section` is the Section instance that contains the call frame info
            while `cfi_entries` must be an iterable that yields the sequence of
            CIE or FDE instances.
        """
    self._emitline('Contents of the %s section:' % section.name)
    for entry in cfi_entries:
        if isinstance(entry, CIE):
            self._emitline('\n%08x %s %s CIE' % (entry.offset, self._format_hex(entry['length'], fullhex=True, lead0x=False), self._format_hex(entry['CIE_id'], fieldsize=8, lead0x=False)))
            self._emitline('  Version:               %d' % entry['version'])
            self._emitline('  Augmentation:          "%s"' % bytes2str(entry['augmentation']))
            self._emitline('  Code alignment factor: %u' % entry['code_alignment_factor'])
            self._emitline('  Data alignment factor: %d' % entry['data_alignment_factor'])
            self._emitline('  Return address column: %d' % entry['return_address_register'])
            if entry.augmentation_bytes:
                self._emitline('  Augmentation data:     {}'.format(' '.join(('{:02x}'.format(ord(b)) for b in iterbytes(entry.augmentation_bytes)))))
            self._emitline()
        elif isinstance(entry, FDE):
            self._emitline('\n%08x %s %s FDE cie=%08x pc=%s..%s' % (entry.offset, self._format_hex(entry['length'], fullhex=True, lead0x=False), self._format_hex(entry['CIE_pointer'], fieldsize=8, lead0x=False), entry.cie.offset, self._format_hex(entry['initial_location'], fullhex=True, lead0x=False), self._format_hex(entry['initial_location'] + entry['address_range'], fullhex=True, lead0x=False)))
            if entry.augmentation_bytes:
                self._emitline('  Augmentation data:     {}'.format(' '.join(('{:02x}'.format(ord(b)) for b in iterbytes(entry.augmentation_bytes)))))
        else:
            assert isinstance(entry, ZERO)
            self._emitline('\n%08x ZERO terminator' % entry.offset)
            continue
        self._emit(describe_CFI_instructions(entry))
    self._emitline()