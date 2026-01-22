from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AnnotationValueValuesEnum(_messages.Enum):
    """Optional. The annotation that will be assigned to the Event. This
    field can be left empty to provide reasons that apply to an event without
    concluding whether the event is legitimate or fraudulent.

    Values:
      ANNOTATION_UNSPECIFIED: Default unspecified type.
      LEGITIMATE: Provides information that the event turned out to be
        legitimate.
      FRAUDULENT: Provides information that the event turned out to be
        fraudulent.
      PASSWORD_CORRECT: Provides information that the event was related to a
        login event in which the user typed the correct password. Deprecated,
        prefer indicating CORRECT_PASSWORD through the reasons field instead.
      PASSWORD_INCORRECT: Provides information that the event was related to a
        login event in which the user typed the incorrect password.
        Deprecated, prefer indicating INCORRECT_PASSWORD through the reasons
        field instead.
    """
    ANNOTATION_UNSPECIFIED = 0
    LEGITIMATE = 1
    FRAUDULENT = 2
    PASSWORD_CORRECT = 3
    PASSWORD_INCORRECT = 4