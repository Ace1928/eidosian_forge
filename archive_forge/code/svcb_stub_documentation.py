from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from dns import rdata
from dns.name import Name
from dns.tokenizer import Tokenizer
Fake module corresponding to dns.rdtypes.IN.SVCB.

  This is needed due to the calling convention of rdata.register_type().
  