from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.type import date_pb2  # type: ignore
class ListObjectsRequest(proto.Message):
    """Request message for ListObjects.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        parent (str):
            Required. Name of the bucket in which to look
            for objects.
        page_size (int):
            Maximum number of ``items`` plus ``prefixes`` to return in a
            single page of responses. As duplicate ``prefixes`` are
            omitted, fewer total results may be returned than requested.
            The service will use this parameter or 1,000 items,
            whichever is smaller.
        page_token (str):
            A previously-returned page token representing
            part of the larger set of results to view.
        delimiter (str):
            If set, returns results in a directory-like mode. ``items``
            will contain only objects whose names, aside from the
            ``prefix``, do not contain ``delimiter``. Objects whose
            names, aside from the ``prefix``, contain ``delimiter`` will
            have their name, truncated after the ``delimiter``, returned
            in ``prefixes``. Duplicate ``prefixes`` are omitted.
        include_trailing_delimiter (bool):
            If true, objects that end in exactly one instance of
            ``delimiter`` will have their metadata included in ``items``
            in addition to ``prefixes``.
        prefix (str):
            Filter results to objects whose names begin
            with this prefix.
        versions (bool):
            If ``true``, lists all versions of an object as distinct
            results. For more information, see `Object
            Versioning <https://cloud.google.com/storage/docs/object-versioning>`__.
        read_mask (google.protobuf.field_mask_pb2.FieldMask):
            Mask specifying which fields to read from each result. If no
            mask is specified, will default to all fields except
            items.acl and items.owner.

            -  may be used to mean "all fields".

            This field is a member of `oneof`_ ``_read_mask``.
        lexicographic_start (str):
            Optional. Filter results to objects whose names are
            lexicographically equal to or after lexicographic_start. If
            lexicographic_end is also set, the objects listed have names
            between lexicographic_start (inclusive) and
            lexicographic_end (exclusive).
        lexicographic_end (str):
            Optional. Filter results to objects whose names are
            lexicographically before lexicographic_end. If
            lexicographic_start is also set, the objects listed have
            names between lexicographic_start (inclusive) and
            lexicographic_end (exclusive).
        soft_deleted (bool):
            Optional. If true, only list all soft-deleted
            versions of the object. Soft delete policy is
            required to set this option.
        include_folders_as_prefixes (bool):
            Optional. If true, will also include folders and managed
            folders (besides objects) in the returned ``prefixes``.
            Requires ``delimiter`` to be set to '/'.
        match_glob (str):
            Optional. Filter results to objects and prefixes that match
            this glob pattern. See `List Objects Using
            Glob <https://cloud.google.com/storage/docs/json_api/v1/objects/list#list-objects-and-prefixes-using-glob>`__
            for the full syntax.
    """
    parent: str = proto.Field(proto.STRING, number=1)
    page_size: int = proto.Field(proto.INT32, number=2)
    page_token: str = proto.Field(proto.STRING, number=3)
    delimiter: str = proto.Field(proto.STRING, number=4)
    include_trailing_delimiter: bool = proto.Field(proto.BOOL, number=5)
    prefix: str = proto.Field(proto.STRING, number=6)
    versions: bool = proto.Field(proto.BOOL, number=7)
    read_mask: field_mask_pb2.FieldMask = proto.Field(proto.MESSAGE, number=8, optional=True, message=field_mask_pb2.FieldMask)
    lexicographic_start: str = proto.Field(proto.STRING, number=10)
    lexicographic_end: str = proto.Field(proto.STRING, number=11)
    soft_deleted: bool = proto.Field(proto.BOOL, number=12)
    include_folders_as_prefixes: bool = proto.Field(proto.BOOL, number=13)
    match_glob: str = proto.Field(proto.STRING, number=14)