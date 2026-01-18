
import sqlite3
import csv
import json
import sys

def connect_to_database(backup_file):
    try:
        conn = sqlite3.connect(backup_file)
        return conn
    except sqlite3.Error as e:
        print(f"Database Error: {e}")
        return None

def extract_messages(conn, contact_number, output_format):
    if conn is None:
        return

    try:
        cursor = conn.cursor()

        query = '''
        SELECT text, date, is_from_me 
        FROM message 
        JOIN handle ON message.handle_id = handle.ROWID 
        WHERE handle.id = ? OR handle.id = ?;
        '''
        cursor.execute(query, (contact_number, format_number(contact_number)))
        messages = cursor.fetchall()

        if not messages:
            print("No messages found with the given contact number.")
            return

        save_messages(messages, output_format)

    except sqlite3.Error as e:
        print(f"Database Error while querying: {e}")
    finally:
        cursor.close()

def save_messages(messages, output_format):
    filename = f'messages.{output_format}'

    if output_format == 'csv':
        save_as_csv(messages, filename)
    elif output_format == 'json':
        save_as_json(messages, filename)
    elif output_format == 'txt':
        save_as_txt(messages, filename)

    print(f"Messages successfully extracted to {filename}.")

def save_as_csv(messages, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Text', 'Date', 'Is_From_Me'])
        writer.writerows(messages)

def save_as_json(messages, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump([{'text': m[0], 'date': m[1], 'is_from_me': m[2]} for m in messages], file, ensure_ascii=False, indent=4)

def save_as_txt(messages, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for msg in messages:
            file.write(f"{msg}\n")

def format_number(number):
    # Add logic to format number to international format
    return number

def main():
    if len(sys.argv) != 4:
        print("Usage: python sms_backup_reader.py <database_file> <contact_number> <output_format>")
        return

    database_file, contact_number, output_format = sys.argv[1], sys.argv[2], sys.argv[3]

    conn = connect_to_database(database_file)
    extract_messages(conn, contact_number, output_format)

if __name__ == "__main__":
    main()
