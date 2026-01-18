"""
title: Email Tool
author: Eidos
date: 2025-01-25
version: 1.0
license: MIT
description: A tool for sending arbitrary emails using SMTP.
requirements: smtplib, email, os, json
"""

import smtplib  # Import the smtplib module for sending emails using SMTP.
from email.mime.text import (
    MIMEText,
)  # Import MIMEText for creating email messages with text content.
from typing import (
    List,
    Dict,
    Any,
)  # Import typing hints for better code readability and type checking.
import os  # Import the os module for operating system related tasks (not directly used in the current selection, but might be used in the broader context).
import json  # Import the json module for working with JSON data (not directly used in the current selection, but might be used in the broader context).
from pydantic import (
    BaseModel,
    Field,
)  # Import BaseModel and Field from pydantic for data validation and settings management.


class Tools:
    """
    A class containing tools, currently focused on email functionalities.
    """

    class Valves(BaseModel):
        """
        Configuration settings for the email tool, managed using Pydantic BaseModel for validation and default values.
        """

        FROM_EMAIL: str = Field(
            default="someone@example.com",
            description="The email a LLM can use",
        )  # Define the sender email address with a default value and description.
        PASSWORD: str = Field(
            default="password",
            description="The password for the provided email address",
        )  # Define the password for the sender email with a default value and description.

    def __init__(self):
        """
        Initializes the Tools class by setting up the default valve configurations.
        """
        self.valves = (
            self.Valves()
        )  # Instantiate the Valves class to load default configurations.

    def get_user_name_and_email_and_id(self, __user__: dict = {}) -> str:
        """
        Extracts and formats user information (name, email, ID) from a user dictionary.

        This method safely accesses user details from the provided dictionary.
        It constructs a user-friendly string containing the user's name, ID, and email if they exist in the dictionary.
        If the user dictionary is empty or lacks user information, it defaults to returning "User: Unknown".

        Args:
            __user__ (dict, optional): A dictionary containing user information. Defaults to an empty dictionary.

        Returns:
            str: A formatted string with user details or "User: Unknown" if no user information is available.
        """

        # The __user__ parameter is intentionally not included in the tool's specification to avoid exposing it directly to LLMs.
        # It is designed to receive the session user object when the function is invoked within the application.

        print(
            __user__
        )  # Print the user object for debugging and logging purposes to inspect the input.
        result = ""  # Initialize an empty string to accumulate user information.

        if (
            "name" in __user__
        ):  # Check if the 'name' key exists in the user dictionary, indicating user name is available.
            result += f"User: {__user__['name']}"  # Append the user's name to the result string.
        if (
            "id" in __user__
        ):  # Check if the 'id' key exists in the user dictionary, indicating user ID is available.
            result += f" (ID: {__user__['id']})"  # Append the user's ID, enclosed in parentheses, to the result string.
        if (
            "email" in __user__
        ):  # Check if the 'email' key exists in the user dictionary, indicating user email is available.
            result += f" (Email: {__user__['email']})"  # Append the user's email, enclosed in parentheses, to the result string.

        if (
            result == ""
        ):  # Check if the result string is still empty after checking for name, ID, and email.
            result = "User: Unknown"  # If no user information was appended, set the result to "User: Unknown".

        return result  # Return the constructed user information string.

    def send_email(self, subject: str, body: str, recipients: List[str]) -> str:
        """
        Sends an email using SMTP with the provided subject, body, and recipient list.

        This method configures and sends an email using the SMTP protocol.
        It retrieves sender credentials from the `valves` configuration.
        It constructs the email message, connects to the SMTP server, authenticates, and sends the email.
        Error handling is included to catch potential issues during the email sending process.

        Important notes for LLMs using this tool:
        - Emails are sent using pre-configured credentials, not on behalf of the current user's email.
        - Always confirm user consent before sending any email.
        - After obtaining consent, proceed to send the email in the subsequent response.

        Args:
            subject (str): The subject line of the email.
            body (str): The main content or body of the email.
            recipients (List[str]): A list of email addresses to send the email to.

        Returns:
            str: A success message including email details if the email is sent successfully,
                 or an error message containing details of any exception encountered.
        """
        sender: str = (
            self.valves.FROM_EMAIL
        )  # Retrieve the sender email address from the configured valves.
        password: str = (
            self.valves.PASSWORD
        )  # Retrieve the password for the sender email address from the configured valves.
        msg = MIMEText(
            body
        )  # Create a MIMEText object to represent the email message, using the provided body.
        msg["Subject"] = subject  # Set the subject of the email message.
        msg["From"] = sender  # Set the sender email address in the email headers.
        msg["To"] = ", ".join(
            recipients
        )  # Set the recipient email addresses in the email headers, joining multiple recipients with commas.

        try:
            with smtplib.SMTP_SSL(
                "smtp.gmail.com", 465
            ) as smtp_server:  # Establish a secure connection to the SMTP server (smtp.gmail.com on port 465) using SSL.
                smtp_server.login(
                    sender, password
                )  # Login to the SMTP server using the configured sender email and password for authentication.
                smtp_server.sendmail(
                    sender, recipients, msg.as_string()
                )  # Send the email message. The msg.as_string() converts the MIMEText object into a string format suitable for sending.
            return f"Message sent:\n   TO: {str(recipients)}\n   SUBJECT: {subject}\n   BODY: {body}"  # Return a success message confirming email delivery and displaying email details.
        except (
            Exception
        ) as e:  # Catch any exceptions that might occur during the email sending process, such as connection errors, authentication failures, etc.
            return str(
                {"status": "error", "message": f"{str(e)}"}
            )  # Return an error message as a JSON string, indicating the status as 'error' and including the exception message for debugging.
