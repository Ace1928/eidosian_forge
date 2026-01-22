from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Citation(_messages.Message):
    """Metadata of one citation.

  Fields:
    endIndex: Index in the prediction output where the citation ends
      (exclusive). Must be > start_index and < len(output).
    license: License associated with this recitation. If present, it refers to
      the license of the source of this citation. Possible licenses include
      code licenses, e.g., mit license.
    publicationDate: Publication date associated with this citation. If
      present, it refers to the date at which the source of this citation was
      published. Possible formats are YYYY, YYYY-MM, YYYY-MM-DD.
    startIndex: Index in the prediction output where the citation starts
      (inclusive). Must be >= 0 and < end_index.
    title: Title associated with this citation. If present, it refers to the
      title of the source of this citation. Possible titles include news
      titles, book titles, etc.
    url: URL associated with this citation. If present, this URL links to the
      webpage of the source of this citation. Possible URLs include news
      websites, GitHub repos, etc.
  """
    endIndex = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    license = _messages.StringField(2)
    publicationDate = _messages.StringField(3)
    startIndex = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    title = _messages.StringField(5)
    url = _messages.StringField(6)