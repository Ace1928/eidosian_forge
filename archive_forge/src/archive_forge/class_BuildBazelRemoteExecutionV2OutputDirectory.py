from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BuildBazelRemoteExecutionV2OutputDirectory(_messages.Message):
    """An `OutputDirectory` is the output in an `ActionResult` corresponding to
  a directory's full contents rather than a single file.

  Fields:
    isTopologicallySorted: If set, consumers MAY make the following
      assumptions about the directories contained in the the Tree, so that it
      may be instantiated on a local file system by scanning through it
      sequentially: - All directories with the same binary representation are
      stored exactly once. - All directories, apart from the root directory,
      are referenced by at least one parent directory. - Directories are
      stored in topological order, with parents being stored before the child.
      The root directory is thus the first to be stored. Additionally, the
      Tree MUST be encoded as a stream of records, where each record has the
      following format: - A tag byte, having one of the following two values:
      - (1 << 3) | 2 == 0x0a: First record (the root directory). - (2 << 3) |
      2 == 0x12: Any subsequent records (child directories). - The size of the
      directory, encoded as a base 128 varint. - The contents of the
      directory, encoded as a binary serialized Protobuf message. This
      encoding is a subset of the Protobuf wire format of the Tree message. As
      it is only permitted to store data associated with field numbers 1 and
      2, the tag MUST be encoded as a single byte. More details on the
      Protobuf wire format can be found here:
      https://developers.google.com/protocol-buffers/docs/encoding It is
      recommended that implementations using this feature construct Tree
      objects manually using the specification given above, as opposed to
      using a Protobuf library to marshal a full Tree message. As individual
      Directory messages already need to be marshaled to compute their
      digests, constructing the Tree object manually avoids redundant
      marshaling.
    path: The full path of the directory relative to the working directory.
      The path separator is a forward slash `/`. Since this is a relative
      path, it MUST NOT begin with a leading forward slash. The empty string
      value is allowed, and it denotes the entire working directory.
    treeDigest: The digest of the encoded Tree proto containing the
      directory's contents.
  """
    isTopologicallySorted = _messages.BooleanField(1)
    path = _messages.StringField(2)
    treeDigest = _messages.MessageField('BuildBazelRemoteExecutionV2Digest', 3)