from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.type import date_pb2  # type: ignore
class Lifecycle(proto.Message):
    """Lifecycle properties of a bucket.
        For more information, see
        https://cloud.google.com/storage/docs/lifecycle.

        Attributes:
            rule (MutableSequence[googlecloudsdk.generated_clients.gapic_clients.storage_v2.types.Bucket.Lifecycle.Rule]):
                A lifecycle management rule, which is made of
                an action to take and the condition(s) under
                which the action will be taken.
        """

    class Rule(proto.Message):
        """A lifecycle Rule, combining an action to take on an object
            and a condition which will trigger that action.

            Attributes:
                action (googlecloudsdk.generated_clients.gapic_clients.storage_v2.types.Bucket.Lifecycle.Rule.Action):
                    The action to take.
                condition (googlecloudsdk.generated_clients.gapic_clients.storage_v2.types.Bucket.Lifecycle.Rule.Condition):
                    The condition(s) under which the action will
                    be taken.
            """

        class Action(proto.Message):
            """An action to take on an object.

                Attributes:
                    type_ (str):
                        Type of the action. Currently, only ``Delete``,
                        ``SetStorageClass``, and ``AbortIncompleteMultipartUpload``
                        are supported.
                    storage_class (str):
                        Target storage class. Required iff the type
                        of the action is SetStorageClass.
                """
            type_: str = proto.Field(proto.STRING, number=1)
            storage_class: str = proto.Field(proto.STRING, number=2)

        class Condition(proto.Message):
            """A condition of an object which triggers some action.

                .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

                Attributes:
                    age_days (int):
                        Age of an object (in days). This condition is
                        satisfied when an object reaches the specified
                        age. A value of 0 indicates that all objects
                        immediately match this condition.

                        This field is a member of `oneof`_ ``_age_days``.
                    created_before (google.type.date_pb2.Date):
                        This condition is satisfied when an object is
                        created before midnight of the specified date in
                        UTC.
                    is_live (bool):
                        Relevant only for versioned objects. If the value is
                        ``true``, this condition matches live objects; if the value
                        is ``false``, it matches archived objects.

                        This field is a member of `oneof`_ ``_is_live``.
                    num_newer_versions (int):
                        Relevant only for versioned objects. If the
                        value is N, this condition is satisfied when
                        there are at least N versions (including the
                        live version) newer than this version of the
                        object.

                        This field is a member of `oneof`_ ``_num_newer_versions``.
                    matches_storage_class (MutableSequence[str]):
                        Objects having any of the storage classes specified by this
                        condition will be matched. Values include
                        ``MULTI_REGIONAL``, ``REGIONAL``, ``NEARLINE``,
                        ``COLDLINE``, ``STANDARD``, and
                        ``DURABLE_REDUCED_AVAILABILITY``.
                    days_since_custom_time (int):
                        Number of days that have elapsed since the
                        custom timestamp set on an object.
                        The value of the field must be a nonnegative
                        integer.

                        This field is a member of `oneof`_ ``_days_since_custom_time``.
                    custom_time_before (google.type.date_pb2.Date):
                        An object matches this condition if the
                        custom timestamp set on the object is before the
                        specified date in UTC.
                    days_since_noncurrent_time (int):
                        This condition is relevant only for versioned
                        objects. An object version satisfies this
                        condition only if these many days have been
                        passed since it became noncurrent. The value of
                        the field must be a nonnegative integer. If it's
                        zero, the object version will become eligible
                        for Lifecycle action as soon as it becomes
                        noncurrent.

                        This field is a member of `oneof`_ ``_days_since_noncurrent_time``.
                    noncurrent_time_before (google.type.date_pb2.Date):
                        This condition is relevant only for versioned
                        objects. An object version satisfies this
                        condition only if it became noncurrent before
                        the specified date in UTC.
                    matches_prefix (MutableSequence[str]):
                        List of object name prefixes. If any prefix
                        exactly matches the beginning of the object
                        name, the condition evaluates to true.
                    matches_suffix (MutableSequence[str]):
                        List of object name suffixes. If any suffix
                        exactly matches the end of the object name, the
                        condition evaluates to true.
                """
            age_days: int = proto.Field(proto.INT32, number=1, optional=True)
            created_before: date_pb2.Date = proto.Field(proto.MESSAGE, number=2, message=date_pb2.Date)
            is_live: bool = proto.Field(proto.BOOL, number=3, optional=True)
            num_newer_versions: int = proto.Field(proto.INT32, number=4, optional=True)
            matches_storage_class: MutableSequence[str] = proto.RepeatedField(proto.STRING, number=5)
            days_since_custom_time: int = proto.Field(proto.INT32, number=7, optional=True)
            custom_time_before: date_pb2.Date = proto.Field(proto.MESSAGE, number=8, message=date_pb2.Date)
            days_since_noncurrent_time: int = proto.Field(proto.INT32, number=9, optional=True)
            noncurrent_time_before: date_pb2.Date = proto.Field(proto.MESSAGE, number=10, message=date_pb2.Date)
            matches_prefix: MutableSequence[str] = proto.RepeatedField(proto.STRING, number=11)
            matches_suffix: MutableSequence[str] = proto.RepeatedField(proto.STRING, number=12)
        action: 'Bucket.Lifecycle.Rule.Action' = proto.Field(proto.MESSAGE, number=1, message='Bucket.Lifecycle.Rule.Action')
        condition: 'Bucket.Lifecycle.Rule.Condition' = proto.Field(proto.MESSAGE, number=2, message='Bucket.Lifecycle.Rule.Condition')
    rule: MutableSequence['Bucket.Lifecycle.Rule'] = proto.RepeatedField(proto.MESSAGE, number=1, message='Bucket.Lifecycle.Rule')