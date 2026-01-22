from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
from . import runtime
class GenericIssueErrorType(enum.Enum):
    CROSS_ORIGIN_PORTAL_POST_MESSAGE_ERROR = 'CrossOriginPortalPostMessageError'
    FORM_LABEL_FOR_NAME_ERROR = 'FormLabelForNameError'
    FORM_DUPLICATE_ID_FOR_INPUT_ERROR = 'FormDuplicateIdForInputError'
    FORM_INPUT_WITH_NO_LABEL_ERROR = 'FormInputWithNoLabelError'
    FORM_AUTOCOMPLETE_ATTRIBUTE_EMPTY_ERROR = 'FormAutocompleteAttributeEmptyError'
    FORM_EMPTY_ID_AND_NAME_ATTRIBUTES_FOR_INPUT_ERROR = 'FormEmptyIdAndNameAttributesForInputError'
    FORM_ARIA_LABELLED_BY_TO_NON_EXISTING_ID = 'FormAriaLabelledByToNonExistingId'
    FORM_INPUT_ASSIGNED_AUTOCOMPLETE_VALUE_TO_ID_OR_NAME_ATTRIBUTE_ERROR = 'FormInputAssignedAutocompleteValueToIdOrNameAttributeError'
    FORM_LABEL_HAS_NEITHER_FOR_NOR_NESTED_INPUT = 'FormLabelHasNeitherForNorNestedInput'
    FORM_LABEL_FOR_MATCHES_NON_EXISTING_ID_ERROR = 'FormLabelForMatchesNonExistingIdError'
    FORM_INPUT_HAS_WRONG_BUT_WELL_INTENDED_AUTOCOMPLETE_VALUE_ERROR = 'FormInputHasWrongButWellIntendedAutocompleteValueError'
    RESPONSE_WAS_BLOCKED_BY_ORB = 'ResponseWasBlockedByORB'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)