
import sqlite3
from tkinter import messagebox, Tk, Label, Entry, Button, StringVar

def extract_messages(conn, contact_number, output_format):
    """
    Extracts messages from the database for a specific contact number.

    Args:
    conn (sqlite3.Connection): The database connection object.
    contact_number (str): The contact number to filter messages.
    output_format (str): The desired output format (e.g., 'csv', 'json').

    Returns:
    list: Extracted messages in the specified format.
    """
    if conn is None:
        return None

    try:
        cursor = conn.cursor()
        query = "SELECT * FROM messages WHERE contact_number = ?"
        cursor.execute(query, (contact_number,))
        messages = cursor.fetchall()
        # Further implementation for different output formats...

        return messages

    except sqlite3.Error as e:
        messagebox.showerror("Extraction Error", f"An error occurred: {e}")
        return None

def create_gui():
    """
    Creates a graphical user interface for the application.
    """
    window = Tk()
    window.title("Extract Messages")

    # GUI widgets for user input and interaction
    Label(window, text="Contact Number").grid(row=0, column=0)
    contact_number_var = StringVar()
    contact_entry = Entry(window, textvariable=contact_number_var)
    contact_entry.grid(row=0, column=1)

    Label(window, text="Output Format").grid(row=1, column=0)
    output_format_var = StringVar()
    output_format_entry = Entry(window, textvariable=output_format_var)
    output_format_entry.grid(row=1, column=1)

    Button(window, text="Extract", command=lambda: extract_messages(None, contact_number_var.get(), output_format_var.get())).grid(row=2, columnspan=2)

    window.mainloop()

# Example usage
if __name__ == "__main__":
    create_gui()
