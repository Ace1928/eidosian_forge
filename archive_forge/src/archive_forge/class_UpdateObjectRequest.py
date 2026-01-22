from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.type import date_pb2  # type: ignore
class UpdateObjectRequest(proto.Message):
    """Request message for UpdateObject.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        object_ (googlecloudsdk.generated_clients.gapic_clients.storage_v2.types.Object):
            Required. The object to update.
            The object's bucket and name fields are used to
            identify the object to update. If present, the
            object's generation field selects a specific
            revision of this object whose metadata should be
            updated. Otherwise, assumes the live version of
            the object.
        if_generation_match (int):
            Makes the operation conditional on whether
            the object's current generation matches the
            given value. Setting to 0 makes the operation
            succeed only if there are no live versions of
            the object.

            This field is a member of `oneof`_ ``_if_generation_match``.
        if_generation_not_match (int):
            Makes the operation conditional on whether
            the object's live generation does not match the
            given value. If no live object exists, the
            precondition fails. Setting to 0 makes the
            operation succeed only if there is a live
            version of the object.

            This field is a member of `oneof`_ ``_if_generation_not_match``.
        if_metageneration_match (int):
            Makes the operation conditional on whether
            the object's current metageneration matches the
            given value.

            This field is a member of `oneof`_ ``_if_metageneration_match``.
        if_metageneration_not_match (int):
            Makes the operation conditional on whether
            the object's current metageneration does not
            match the given value.

            This field is a member of `oneof`_ ``_if_metageneration_not_match``.
        predefined_acl (str):
            Apply a predefined set of access controls to
            this object. Valid values are
            "authenticatedRead", "bucketOwnerFullControl",
            "bucketOwnerRead", "private", "projectPrivate",
            or "publicRead".
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Required. List of fields to be updated.

            To specify ALL fields, equivalent to the JSON API's "update"
            function, specify a single field with the value ``*``. Note:
            not recommended. If a new field is introduced at a later
            time, an older client updating with the ``*`` may
            accidentally reset the new field's value.

            Not specifying any fields is an error.
        common_object_request_params (googlecloudsdk.generated_clients.gapic_clients.storage_v2.types.CommonObjectRequestParams):
            A set of parameters common to Storage API
            requests concerning an object.
    """
    object_: 'Object' = proto.Field(proto.MESSAGE, number=1, message='Object')
    if_generation_match: int = proto.Field(proto.INT64, number=2, optional=True)
    if_generation_not_match: int = proto.Field(proto.INT64, number=3, optional=True)
    if_metageneration_match: int = proto.Field(proto.INT64, number=4, optional=True)
    if_metageneration_not_match: int = proto.Field(proto.INT64, number=5, optional=True)
    predefined_acl: str = proto.Field(proto.STRING, number=10)
    update_mask: field_mask_pb2.FieldMask = proto.Field(proto.MESSAGE, number=7, message=field_mask_pb2.FieldMask)
    common_object_request_params: 'CommonObjectRequestParams' = proto.Field(proto.MESSAGE, number=8, message='CommonObjectRequestParams')