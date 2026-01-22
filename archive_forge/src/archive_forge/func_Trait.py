from types import FunctionType, MethodType
import warnings
from .constants import (
from .ctrait import CTrait
from .trait_errors import TraitError
from .trait_base import (
from .trait_converters import (
from .trait_handler import TraitHandler
from .trait_type import (
from .trait_handlers import (
from .trait_factory import (
from .util.deprecated import deprecated
def Trait(*value_type, **metadata):
    """ Creates a trait definition.

    This function accepts a variety of forms of parameter lists:

    +-------------------+---------------+-------------------------------------+
    | Format            | Example       | Description                         |
    +===================+===============+=====================================+
    | Trait(*default*)  | Trait(150.0)  | The type of the trait is inferred   |
    |                   |               | from the type of the default value, |
    |                   |               | which must be in *ConstantTypes*.   |
    +-------------------+---------------+-------------------------------------+
    | Trait(*default*,  | Trait(None,   | The trait accepts any of the        |
    | *other1*,         | 0, 1, 2,      | enumerated values, with the first   |
    | *other2*, ...)    | 'many')       | value being the default value. The  |
    |                   |               | values must be of types in          |
    |                   |               | *ConstantTypes*, but they need not  |
    |                   |               | be of the same type. The *default*  |
    |                   |               | value is not valid for assignment   |
    |                   |               | unless it is repeated later in the  |
    |                   |               | list.                               |
    +-------------------+---------------+-------------------------------------+
    | Trait([*default*, | Trait([None,  | Similar to the previous format, but |
    | *other1*,         | 0, 1, 2,      | takes an explicit list or a list    |
    | *other2*, ...])   | 'many'])      | variable.                           |
    +-------------------+---------------+-------------------------------------+
    | Trait(*type*)     | Trait(Int)    | The *type* parameter must be a name |
    |                   |               | of a Python type (see               |
    |                   |               | *PythonTypes*). Assigned values     |
    |                   |               | must be of exactly the specified    |
    |                   |               | type; no casting or coercion is     |
    |                   |               | performed. The default value is the |
    |                   |               | appropriate form of zero, False,    |
    |                   |               | or emtpy string, set or sequence.   |
    +-------------------+---------------+-------------------------------------+
    | Trait(*class*)    |::             | Values must be instances of *class* |
    |                   |               | or of a subclass of *class*. The    |
    |                   | class MyClass:| default value is None, but None     |
    |                   |    pass       | cannot be assigned as a value.      |
    |                   | foo = Trait(  |                                     |
    |                   | MyClass)      |                                     |
    +-------------------+---------------+-------------------------------------+
    | Trait(None,       |::             | Similar to the previous format, but |
    | *class*)          |               | None *can* be assigned as a value.  |
    |                   | class MyClass:|                                     |
    |                   |   pass        |                                     |
    |                   | foo = Trait(  |                                     |
    |                   | None, MyClass)|                                     |
    +-------------------+---------------+-------------------------------------+
    | Trait(*instance*) |::             | Values must be instances of the     |
    |                   |               | same class as *instance*, or of a   |
    |                   | class MyClass:| subclass of that class. The         |
    |                   |    pass       | specified instance is the default   |
    |                   | i = MyClass() | value.                              |
    |                   | foo =         |                                     |
    |                   |   Trait(i)    |                                     |
    +-------------------+---------------+-------------------------------------+
    | Trait(*handler*)  | Trait(        | Assignment to this trait is         |
    |                   | TraitEnum )   | validated by an object derived from |
    |                   |               | **traits.TraitHandler**.            |
    +-------------------+---------------+-------------------------------------+
    | Trait(*default*,  | Trait(0.0, 0.0| This is the most general form of    |
    | { *type* |        | 'stuff',      | the function. The notation:         |
    | *constant* |      | TupleType)    | ``{...|...|...}+`` means a list of  |
    | *dict* | *class* ||               | one or more of any of the items     |
    | *function* |      |               | listed between the braces. Thus, the|
    | *handler* |       |               | most general form of the function   |
    | *trait* }+ )      |               | consists of a default value,        |
    |                   |               | followed by one or more of several  |
    |                   |               | possible items. A trait defined by  |
    |                   |               | multiple items is called a          |
    |                   |               | "compound" trait.                   |
    +-------------------+---------------+-------------------------------------+

    All forms of the Trait function accept both predefined and arbitrary
    keyword arguments. The value of each keyword argument becomes bound to the
    resulting trait object as the value of an attribute having the same name
    as the keyword. This feature lets you associate metadata with a trait.

    The following predefined keywords are accepted:

    desc : str
        Describes the intended meaning of the trait. It is used in
        exception messages and fly-over help in user interfaces.
    label : str
        Provides a human-readable name for the trait. It is used to label user
        interface editors for traits.
    editor : traits.api.Editor
        Instance of a subclass Editor object to use when creating a user
        interface editor for the trait. See the "Traits UI User Guide" for
        more information on trait editors.
    comparison_mode : int
        Indicates when trait change notifications should be generated based
        upon the result of comparing the old and new values of a trait
        assignment. Possible values come from the ``ComparisonMode`` enum:

        * 0 (none): The values are not compared and a trait change
          notification is generated on each assignment.
        * 1 (identity): A trait change notification is
          generated if the old and new values are not the same object.
        * 2 (equality): A trait change notification is generated if the
          old and new values are not equal using Python's standard equality
          testing. This is the default.

    """
    return _TraitMaker(*value_type, **metadata).as_ctrait()