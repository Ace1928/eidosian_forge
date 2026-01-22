from heat.common import exception
from heat.common.i18n import _
from heat.common import password_gen
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
from heat.engine import translation
def _generate_random_string(self, char_sequences, char_classes, length):
    seq_mins = [password_gen.special_char_class(char_seq[self.CHARACTER_SEQUENCES_SEQUENCE], char_seq[self.CHARACTER_SEQUENCES_MIN]) for char_seq in char_sequences]
    char_class_mins = [password_gen.named_char_class(char_class[self.CHARACTER_CLASSES_CLASS], char_class[self.CHARACTER_CLASSES_MIN]) for char_class in char_classes]
    return password_gen.generate_password(length, seq_mins + char_class_mins)