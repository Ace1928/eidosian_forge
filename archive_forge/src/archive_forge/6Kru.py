import openai  # Import the OpenAI library to use the OpenAI API
from pathlib import Path  # Import the Path class to work with file paths
import json  # Import the json module for working with JSON data
from typing import Dict, Tuple  # Import typing for better type annotations
import duckdb  # Import DuckDB for database operations
import streamlit as st  # Import the Streamlit library for building the user interface
from openai import OpenAI  # For client creation
from openai.types.beta.threads import (
    Annotation,
    AnnotationDelta,
    FileCitationAnnotation,
    FileCitationDeltaAnnotation,
    FilePathAnnotation,
    FilePathDeltaAnnotation,
    ImageFile,
    ImageFileContentBlock,
    ImageFileDelta,
    ImageFileDeltaBlock,
    ImageURL,
    ImageURLContentBlock,
    ImageURLDelta,
    ImageURLDeltaBlock,
    Message,
    MessageContent,
    MessageContentDelta,
    MessageContentPartParam,
    MessageDeleted,
    MessageDelta,
    MessageDeltaEvent,
    Text,
    TextContentBlock,
    TextContentBlockParam,
    TextDelta,
    TextDeltaBlock,
)
from openai.types.beta.threads.runs import (
    CodeInterpreterLogs,
    CodeInterpreterOutputImage,
    CodeInterpreterToolCall,
    CodeInterpreterToolCallDelta,
    FileSearchToolCall,
    FileSearchToolCallDelta,
    FunctionToolCall,
    FunctionToolCallDelta,
    MessageCreationStepDetails,
    RunStep,
    RunStepDelta,
    RunStepDeltaEvent,
    RunStepDeltaMessageDelta,
    ToolCall,
    ToolCallDelta,
    ToolCallDeltaObject,
    ToolCallsStepDetails,
)
from openai.types.beta.threads import RequiredActionFunctionToolCall, Run, RunStatus
from openai.types.beta import (
    AssistantResponseFormat,
    AssistantResponseFormatOption,
    AssistantToolChoice,
    AssistantToolChoiceFunction,
    AssistantToolChoiceOption,
    Thread,
    ThreadDeleted,
)
from openai.types.beta import (
    Assistant,
    AssistantDeleted,
    AssistantStreamEvent,
    AssistantTool,
    CodeInterpreterTool,
    FileSearchTool,
    FunctionTool,
    MessageStreamEvent,
    RunStepStreamEvent,
    RunStreamEvent,
    ThreadStreamEvent,
)

# Constants for OpenAI credentials and settings
ORGANIZATION_ID = "org-rUwFIGjF81JyxeoAEY0ySqdY"
PROJECT_ID = "proj_CEdm0URq3VlXnLt1TO71Uivx"
API_KEY = "sk-proj-iNW4K9fhR5ybSzDw2021T3BlbkFJsthE0eZOEtGELgpsRy0d"
DATABASE_FILE = "chat_database.db"  # DuckDB database file

# Initialize the DuckDB database
conn = duckdb.connect(DATABASE_FILE)

# Ensure the necessary tables exist with appropriate settings
conn.execute(
    """
CREATE TABLE IF NOT EXISTS configuration (
    thread_id VARCHAR,
    assistant_id VARCHAR
)
"""
)
conn.execute(
    """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER,
    type VARCHAR,
    content TEXT,
    PRIMARY KEY (id)
)
"""
)
conn.execute(
    """
CREATE SEQUENCE IF NOT EXISTS messages_id_seq;
"""
)
conn.execute(
    """
ALTER TABLE messages ALTER COLUMN id SET DEFAULT NEXTVAL('messages_id_seq');
"""
)
conn.execute(
    """
CREATE TABLE IF NOT EXISTS model_data (
    user_input TEXT,
    assistant_response TEXT
)
"""
)


# Function to create OpenAI client
def create_client() -> openai.OpenAI:
    """
    Creates and returns an OpenAI client instance using the provided credentials.
    """
    return OpenAI(
        organization=ORGANIZATION_ID,
        project=PROJECT_ID,
        api_key=API_KEY,
    )


# Function to create an assistant
def create_assistant(client: openai.OpenAI) -> Dict:
    """
    Creates and returns an assistant instance using the provided OpenAI client.
    """
    assistant = client.beta.assistants.create(
        name="Indego",
        instructions="You are a precursor version of INDEGO. Your goal is to assist Lloyd Handyside with AI development and motivation.",
        tools=[{"type": "code_interpreter"}],
        model="gpt-4",
    )
    # Save assistant details to the database
    conn.execute("INSERT INTO configuration (assistant_id) VALUES (?)", (assistant.id,))
    return assistant


# Function to create a thread
def create_thread(client: openai.OpenAI) -> Dict:
    """
    Creates and returns a new thread instance using the provided OpenAI client.
    """
    thread = client.beta.threads.create()
    # Save thread details to the database
    conn.execute("INSERT INTO configuration (thread_id) VALUES (?)", (thread.id,))
    return thread


# Function to send a message to the assistant and handle streaming response
def send_message(client: openai.OpenAI, thread_id: str, content: str) -> Dict:
    """
    Sends a message to the assistant using the provided OpenAI client, thread ID, and message content.
    Handles streaming response and returns the created message instance.
    """
    # Log user message to the database
    conn.execute(
        "INSERT INTO messages (type, content) VALUES (?, ?)", ("user", content)
    )

    # Send message to assistant and handle streaming response
    assistant_response = ""
    for event in client.beta.threads.create_and_run_stream(
        thread_id=thread_id,
        messages=[{"role": "user", "content": content}],
    ):
        if isinstance(event, AssistantStreamEvent):
            if isinstance(event.delta):
                if isinstance(event.delta.message_delta):
                    assistant_response += event.delta.message_delta.text
                    st.text_area(
                        "Assistant Response",
                        value=assistant_response,
                        height=200,
                        key="assistant_response",
                    )

    # Log assistant response to the database
    conn.execute(
        "INSERT INTO messages (type, content) VALUES (?, ?)",
        ("assistant", assistant_response),
    )

    return {"role": "assistant", "content": assistant_response}


# Streamlit and logging utilities
def setup_streamlit():
    """
    Sets up the Streamlit UI components.
    """
    st.title("Indego Alpha Chat")


def display_messages():
    """
    Fetches and displays all messages from the database in a scrollable text area in the Streamlit interface.
    """
    messages = conn.execute("SELECT type, content FROM messages ORDER BY id").fetchall()
    chat_log = "\n".join([f"{msg[0]}: {msg[1]}" for msg in messages])
    st.text_area("Chat Log", value=chat_log, height=300, key="log")


# Entry point of the script
if __name__ == "__main__":
    setup_streamlit()
    client = create_client()
    assistant = create_assistant(client)
    thread = create_thread(client)

    # Input field for user to interact with the assistant
    user_input = st.text_input(
        "Introduce yourself to Indego and request assistance.", ""
    )
    if user_input:
        send_message(client, thread.id, user_input)
        display_messages()  # Display all messages dynamically
