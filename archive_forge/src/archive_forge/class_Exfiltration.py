from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Exfiltration(_messages.Message):
    """Exfiltration represents a data exfiltration attempt from one or more
  sources to one or more targets. The `sources` attribute lists the sources of
  the exfiltrated data. The `targets` attribute lists the destinations the
  data was copied to.

  Fields:
    sources: If there are multiple sources, then the data is considered
      "joined" between them. For instance, BigQuery can join multiple tables,
      and each table would be considered a source.
    targets: If there are multiple targets, each target would get a complete
      copy of the "joined" source data.
    totalExfiltratedBytes: Total exfiltrated bytes processed for the entire
      job.
  """
    sources = _messages.MessageField('ExfilResource', 1, repeated=True)
    targets = _messages.MessageField('ExfilResource', 2, repeated=True)
    totalExfiltratedBytes = _messages.IntegerField(3)