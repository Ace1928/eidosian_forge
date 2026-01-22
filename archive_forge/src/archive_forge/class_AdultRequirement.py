class AdultRequirement(Requirement):
    """
    Requires workers to acknowledge that they are over 18 and that they agree to work on potentially offensive content. The value type is boolean, 1 (required), 0 (not required, the default).
    """

    def __init__(self, comparator, integer_value, required_to_preview=False):
        super(AdultRequirement, self).__init__(qualification_type_id='00000000000000000060', comparator=comparator, integer_value=integer_value, required_to_preview=required_to_preview)