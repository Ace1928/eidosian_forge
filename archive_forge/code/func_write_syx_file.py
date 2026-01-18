import re
from .parser import Parser
def write_syx_file(filename, messages, plaintext=False):
    """Write sysex messages to a SYX file.

    Messages other than sysex will be skipped.

    By default this will write the binary format.  Pass
    ``plaintext=True`` to write the plain text format (hex encoded
    ASCII text).
    """
    messages = [m for m in messages if m.type == 'sysex']
    if plaintext:
        with open(filename, 'w') as outfile:
            for message in messages:
                outfile.write(message.hex())
                outfile.write('\n')
    else:
        with open(filename, 'wb') as outfile:
            for message in messages:
                outfile.write(message.bin())