from textwrap import dedent
from ..argument import Argument
from ..enum import Enum, PyEnum
from ..field import Field
from ..inputfield import InputField
from ..inputobjecttype import InputObjectType
from ..mutation import Mutation
from ..scalars import String
from ..schema import ObjectType, Schema
class CreatePaint(Mutation):

    class Arguments:
        color_input = ColorInput(required=True)
    color = RGB(required=True)

    def mutate(_, info, color_input):
        nonlocal color_input_value
        color_input_value = color_input.color
        return CreatePaint(color=color_input.color)