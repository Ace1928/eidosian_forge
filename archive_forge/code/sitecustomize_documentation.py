import sys

    This module will:
    - change the input() and raw_input() commands to change 
 or  into 

    - execute the user site customize -- if available
    - change raw_input() and input() to also remove any trailing 

    Up to PyDev 3.4 it also was setting the default encoding, but it was removed because of differences when
    running from a shell (i.e.: now we just set the PYTHONIOENCODING related to that -- which is properly
    treated on Py 2.7 onwards).
