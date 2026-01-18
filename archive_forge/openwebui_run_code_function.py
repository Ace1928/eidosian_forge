from pydantic import BaseModel, Field
from typing import Optional
from fastapi.requests import (
    Request,
)  # Although Request is imported, it's not used in the current selection. It might be used elsewhere or intended for future use.
from io import (
    StringIO,
)  # Used to redirect stdout to capture the output of executed code.
import sys  # Used to manipulate stdout.

"""
MODERATOR COMMENT: This function should be utilized with EXTREME caution.
Do not expose to untrusted users or deploy on secure networks unless you are sure you have considered all risks.
"""


class Action:
    """
    Action class to execute Python code and manage related functionalities.
    This class is designed to handle actions, specifically executing Python code snippets provided as input.
    It includes features for capturing the output of the executed code and providing status updates via an event emitter.
    """

    class Valves(BaseModel):
        """
        Valves class (currently empty).
        This class is intended to represent configurable valves for the Action, but is currently not in use.
        It is defined as a Pydantic BaseModel for potential future extensions and configurations.
        """

        pass  # Currently no valves are defined.

    class UserValves(BaseModel):
        """
        UserValves class to manage user-specific valve settings.
        This class defines settings that can be customized on a per-user basis, affecting the behavior of the Action.
        Currently, it includes a setting to control the display of status updates.
        """

        show_status: bool = Field(
            default=True, description="Show status of the action."
        )  # Field to control whether to show status updates. Defaults to True.
        pass  # Add more user-specific valves here in the future.

    def __init__(self):
        """
        Initializes the Action class.
        Sets up the default valves for the action. Currently, it initializes the 'valves' attribute with an instance of the empty Valves class.
        """
        self.valves = (
            self.Valves()
        )  # Initializes the valves attribute with an instance of Valves.
        pass  # Add any other initialization logic here.

    def execute_python_code(self, code: str) -> str:
        """
        Executes Python code and returns the output as a string.
        This method takes a string containing Python code, redirects the standard output to capture the execution result,
        executes the code in a controlled environment (empty namespace), and returns the captured output.
        If any exception occurs during execution, it catches the exception and returns an error message.

        Args:
            code (str): The Python code to be executed.

        Returns:
            str: The output of the executed code, or an error message if execution fails.
        """
        # Store the original standard output to restore it later.
        old_stdout = sys.stdout
        # Redirect standard output to StringIO to capture it.
        redirected_output = sys.stdout = StringIO()
        try:
            # Execute the provided Python code in a new empty namespace (dictionary).
            exec(code, {})
        except Exception as e:
            # Restore the original standard output in case of error.
            sys.stdout = old_stdout
            # Return an error message including the exception details.
            return f"Error: {str(e)}"
        finally:
            # Ensure the original standard output is restored, regardless of success or failure.
            sys.stdout = old_stdout
        # Return the captured output from the executed code as a string.
        return redirected_output.getvalue()

    async def action(
        self,
        body: dict,
        __user__=None,
        __event_emitter__=None,
        __event_call__=None,
    ) -> Optional[dict]:
        """
        Asynchronous action method to process requests, detect Python code, and execute it.
        This is the main asynchronous method that handles incoming requests (body), user context (__user__),
        event emission (__event_emitter__), and event calls (__event_call__).
        It checks for Python code blocks in the last message content, executes the code, and returns the result.
        It also manages status updates via the event emitter based on user valve settings.

        Args:
            body (dict): The request body containing messages and other relevant data.
            __user__ (Optional[dict]): User-specific context, potentially containing valve settings. Can be None.
            __event_emitter__ (Optional[callable]): Asynchronous function to emit events for status updates. Can be None.
            __event_call__ (Optional[callable]):  Optional callback function for event handling (currently not used in this method).

        Returns:
            Optional[dict]: A dictionary containing the result of code execution, or None if no code is executed or an error occurs.
                           Returns None if no action is taken that requires a return value, especially when no code is detected or executed.
        """
        # Print the action name for logging or debugging purposes.
        print(f"action:{__name__}")

        # Initialize user_valves to None.
        user_valves = None
        # Check if __user__ is not None before accessing its attributes to avoid potential errors.
        if __user__ is not None:
            # Try to get user valves from the user context.
            user_valves = __user__.get("valves")

        # If user_valves is not found in user context or if __user__ was None.
        if not user_valves:
            # Fallback to default UserValves if no user-specific valves are provided.
            user_valves = self.UserValves()

        # Check if an event emitter is provided before attempting to use it.
        if __event_emitter__:
            # Get the last message from the messages array in the body.
            last_assistant_message = body["messages"][-1]

            # Check if status updates are enabled in user valves.
            if user_valves.show_status:
                # Emit a status update event to indicate processing started.
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Processing your input", "done": False},
                    }
                )

            # Extract the content of the last assistant message.
            input_text = last_assistant_message["content"]
            # Check if the input text is enclosed in ```python and ``` markers.
            if input_text.startswith("```python") and input_text.endswith("```"):
                # Remove the ```python and ``` markers and any leading/trailing whitespace to extract the code.
                code = input_text[9:-3].strip()
                # Execute the extracted Python code using the execute_python_code method.
                output = self.execute_python_code(code)
                # Return the result of code execution in a structured dictionary.
                return {
                    "type": "code_execution_result",
                    "data": {"output": output},
                }

            # Check again if status updates are enabled before sending the 'no code detected' status.
            if user_valves.show_status:
                # Emit a status update event to indicate no valid Python code was detected and processing is done.
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "No valid Python code detected",
                            "done": True,
                        },
                    }
                )
        # Return None if no code was executed and no result to return. Explicitly return None to satisfy the Optional[dict] return type and address the 'Missing return statement' lint.
        return None
