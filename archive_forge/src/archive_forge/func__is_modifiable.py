from dissononce.processing.modifiers.patternmodifier import PatternModifier
def _is_modifiable(self, handsakepattern):
    return handsakepattern.message_patterns[0] in self.__class__.VALID_FIRST_MESSAGES