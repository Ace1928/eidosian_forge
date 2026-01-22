from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Recipe(_messages.Message):
    """Steps taken to build the artifact. For a TaskRun, typically each
  container corresponds to one step in the recipe.

  Messages:
    ArgumentsValueListEntry: A ArgumentsValueListEntry object.
    EnvironmentValueListEntry: A EnvironmentValueListEntry object.

  Fields:
    arguments: Collection of all external inputs that influenced the build on
      top of recipe.definedInMaterial and recipe.entryPoint. For example, if
      the recipe type were "make", then this might be the flags passed to make
      aside from the target, which is captured in recipe.entryPoint. Since the
      arguments field can greatly vary in structure, depending on the builder
      and recipe type, this is of form "Any".
    definedInMaterial: Index in materials containing the recipe steps that are
      not implied by recipe.type. For example, if the recipe type were "make",
      then this would point to the source containing the Makefile, not the
      make program itself. Set to -1 if the recipe doesn't come from a
      material, as zero is default unset value for int64.
    entryPoint: String identifying the entry point into the build. This is
      often a path to a configuration file and/or a target label within that
      file. The syntax and meaning are defined by recipe.type. For example, if
      the recipe type were "make", then this would reference the directory in
      which to run make as well as which target to use.
    environment: Any other builder-controlled inputs necessary for correctly
      evaluating the recipe. Usually only needed for reproducing the build but
      not evaluated as part of policy. Since the environment field can greatly
      vary in structure, depending on the builder and recipe type, this is of
      form "Any".
    type: URI indicating what type of recipe was performed. It determines the
      meaning of recipe.entryPoint, recipe.arguments, recipe.environment, and
      materials.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ArgumentsValueListEntry(_messages.Message):
        """A ArgumentsValueListEntry object.

    Messages:
      AdditionalProperty: An additional property for a ArgumentsValueListEntry
        object.

    Fields:
      additionalProperties: Properties of the object. Contains field @type
        with type URL.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ArgumentsValueListEntry object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class EnvironmentValueListEntry(_messages.Message):
        """A EnvironmentValueListEntry object.

    Messages:
      AdditionalProperty: An additional property for a
        EnvironmentValueListEntry object.

    Fields:
      additionalProperties: Properties of the object. Contains field @type
        with type URL.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a EnvironmentValueListEntry object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    arguments = _messages.MessageField('ArgumentsValueListEntry', 1, repeated=True)
    definedInMaterial = _messages.IntegerField(2)
    entryPoint = _messages.StringField(3)
    environment = _messages.MessageField('EnvironmentValueListEntry', 4, repeated=True)
    type = _messages.StringField(5)