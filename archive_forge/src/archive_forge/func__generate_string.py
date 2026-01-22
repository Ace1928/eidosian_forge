import random
import uuid
def _generate_string(sep, integer_scale):
    predicate = random.choice(_GENERATOR_PREDICATES).lower()
    noun = random.choice(_GENERATOR_NOUNS).lower()
    num = random.randint(0, 10 ** integer_scale)
    return f'{predicate}{sep}{noun}{sep}{num}'