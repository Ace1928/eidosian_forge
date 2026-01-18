from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import random
import string
Generate a random string of 3-letter sections.

  Each word has a 1/2205 chance of being generated (see _GenerateThreeLetter).
  Therefore a specific name has a (1/2205)^(sections) chance of being generated.
  For 3 sections, the denominator is over 10.7 billion.

  Args:
    sections: int, number of 3-letter generated sections to include
    separator: str, separator between sections
    prefix: str, prefix of the generated name. This acts like an additional
      section at the start of the name and will be separated from the
      generated sections by the seperator argument, however it does not count
      towards the number of sections specified by the sections argument.
    validate: bool, True to validate sections against invalid word list

  Returns:
    str, generated name
  