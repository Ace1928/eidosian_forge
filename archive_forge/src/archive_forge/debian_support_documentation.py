import os
import os.path
import re
from debian.deprecation import function_deprecated_by
import debian._arch_table
Creates a new package file object.

        name - the name of the file the data comes from
        file_obj - an alternate data source; the default is to open the
                  file with the indicated name.
        