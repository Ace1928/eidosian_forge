import os
import pytest

    Run tests to ensure that the VASP txt and label arguments function correctly,
    i.e. correctly sets the working directories and works in that directory.

    This is conditional on the existence of the ASE_VASP_COMMAND, VASP_COMMAND
    or VASP_SCRIPT environment variables

    