from __future__ import annotations
import re
@staticmethod
def validate_verbose(labels: dict[str, str]) -> tuple[bool, str]:
    """Validates Labels and returns the corresponding error message if something is wrong. Returns True, <empty string>
        if everything is fine.

        :return:  bool, str
        """
    for key, value in labels.items():
        if LabelValidator.KEY_REGEX.match(key) is None:
            return (False, f'label key {key} is not correctly formatted')
        if LabelValidator.VALUE_REGEX.match(value) is None:
            return (False, f'label value {value} (key: {key}) is not correctly formatted')
    return (True, '')