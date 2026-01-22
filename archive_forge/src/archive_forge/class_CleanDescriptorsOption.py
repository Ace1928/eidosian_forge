from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CleanDescriptorsOption(_messages.Message):
    """This option is based on the DICOM Standard's [Clean Descriptors Option](
  http://dicom.nema.org/medical/dicom/2018e/output/chtml/part15/sect_E.3.5.htm
  l), and the `CleanText` `Action` is applied to all the specified fields.
  When cleaning text, the process attempts to transform phrases matching any
  of the tags marked for removal (action codes D, Z, X, and U) in the [Basic P
  rofile](http://dicom.nema.org/medical/dicom/2018e/output/chtml/part15/chapte
  r_E.html). These contextual phrases are replaced with the token "[CTX]".
  This option uses an additional infoType during inspection.
  """