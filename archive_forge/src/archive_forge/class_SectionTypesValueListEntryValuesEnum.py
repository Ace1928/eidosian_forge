from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SectionTypesValueListEntryValuesEnum(_messages.Enum):
    """SectionTypesValueListEntryValuesEnum enum type.

    Values:
      SECTION_TYPE_UNSPECIFIED: Undefined section type, does not return
        anything.
      SITUATION: What the customer needs help with or has question about.
        Section name: "situation".
      ACTION: What the agent does to help the customer. Section name:
        "action".
      RESOLUTION: Result of the customer service. A single word describing the
        result of the conversation. Section name: "resolution".
      REASON_FOR_CANCELLATION: Reason for cancellation if the customer
        requests for a cancellation. "N/A" otherwise. Section name:
        "reason_for_cancellation".
      CUSTOMER_SATISFACTION: "Unsatisfied" or "Satisfied" depending on the
        customer's feelings at the end of the conversation. Section name:
        "customer_satisfaction".
      ENTITIES: Key entities extracted from the conversation, such as ticket
        number, order number, dollar amount, etc. Section names are prefixed
        by "entities/".
    """
    SECTION_TYPE_UNSPECIFIED = 0
    SITUATION = 1
    ACTION = 2
    RESOLUTION = 3
    REASON_FOR_CANCELLATION = 4
    CUSTOMER_SATISFACTION = 5
    ENTITIES = 6