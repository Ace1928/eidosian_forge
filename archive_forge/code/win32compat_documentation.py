import os
import pathlib
import platform
import stat
import sys
from logging import getLogger
from typing import Union
Implementing the below in Python:

        typedef struct _REPARSE_DATA_BUFFER {
            ULONG  ReparseTag;
            USHORT ReparseDataLength;
            USHORT Reserved;
            union {
                struct {
                    USHORT SubstituteNameOffset;
                    USHORT SubstituteNameLength;
                    USHORT PrintNameOffset;
                    USHORT PrintNameLength;
                    ULONG Flags;
                    WCHAR PathBuffer[1];
                } SymbolicLinkReparseBuffer;
                struct {
                    USHORT SubstituteNameOffset;
                    USHORT SubstituteNameLength;
                    USHORT PrintNameOffset;
                    USHORT PrintNameLength;
                    WCHAR PathBuffer[1];
                } MountPointReparseBuffer;
                struct {
                    UCHAR  DataBuffer[1];
                } GenericReparseBuffer;
            } DUMMYUNIONNAME;
        } REPARSE_DATA_BUFFER, *PREPARSE_DATA_BUFFER;
        