import argparse
import sys
import mido
from mido import Message, MidiFile, tempo2bpm

Play MIDI file on output port.

Example:

    mido-play some_file.mid

Todo:

  - add option for printing messages
