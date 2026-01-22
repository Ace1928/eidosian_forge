from pydantic import BaseModel, create_model, ValidationError
from typing import Dict, Type, Any, Tuple, Union, TypeVar

# Type Aliases for clarity and precision
FieldsDictType = Dict[
    str, Tuple[Type[Any], Union[Any, Tuple[()]]]
]  # Defines the structure of fields dictionary
ModelType = TypeVar(
    "ModelType", bound=BaseModel
)  # Type variable bound to BaseModel for return type annotation


class UniversalModel(BaseModel):
    """
    A base class for a universal model that can dynamically evolve over time.
    This class serves as the foundation for creating models on-the-fly based on runtime data,
    allowing for the dynamic extension and adaptation of data models in response to changing requirements.
    """

    @classmethod
    def create_dynamic_model(cls, name: str, fields: FieldsDictType) -> ModelType:
        """
        Dynamically creates a new model extending the current one with additional fields.

        Args:
            name (str): The name of the new model class.
            fields (FieldsDictType): A dictionary where keys are field names and values are tuples of (type, default value).

        Returns:
            ModelType: A new Pydantic model class with the specified fields added.

        Raises:
            ValueError: If any of the new fields clash with existing fields in the base model.
        """
        # Ensure the new fields do not clash with existing fields
        existing_fields = cls.__fields__.keys()
        for field in fields.keys():
            if field in existing_fields:
                raise ValueError(f"Field '{field}' already exists in the base model.")

        # Correctly structuring explicit_fields for Pydantic
        explicit_fields = {
            field: (field_type, field_default)
            for field, (field_type, field_default) in fields.items()
        }

        # Create the model with the structured fields
        new_model: ModelType = create_model(
            name, __base__=cls, **explicit_fields
        )  # Type casting to ModelType
        return new_model


# Example usage
if __name__ == "__main__":
    # Define initial fields for the dynamic model
    initial_fields: FieldsDictType = {
        "age": (int, ...),  # Field 'age' of type int is required
        "name": (str, None),  # Field 'name' of type str is optional
    }

    # Create a dynamic model based on the initial fields
    DynamicUserModel: ModelType = UniversalModel.create_dynamic_model(
        "DynamicUserModel", initial_fields
    )

    # Example of using the dynamically created model
    try:
        user_instance = DynamicUserModel(age=30, name="John Doe")
        print(user_instance.json())
    except ValidationError as e:
        print(f"Validation error occurred: {e}")

    # Dynamically extend the model with new fields
    extended_fields: FieldsDictType = {
        "email": (str, ...),  # Adding a new required field 'email'
    }
    try:
        ExtendedUserModel: ModelType = DynamicUserModel.create_dynamic_model(
            "ExtendedUserModel", extended_fields
        )

        # Example of using the extended model
        extended_user_instance = ExtendedUserModel(
            age=30, name="Jane Doe", email="jane@example.com"
        )
        print(extended_user_instance.json())
    except ValidationError as e:
        print(f"Validation error occurred: {e}")
