import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

sender_email = "your_email@example.com"
sender_password = "your_email_password"

recipients = [
    {"name": "John Doe", "email": "john@example.com"},
    {"name": "Jane Smith", "email": "jane@example.com"},
    {"name": "Mike Johnson", "email": "mike@example.com"},
]

for recipient in recipients:
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = recipient["email"]
    message["Subject"] = "Personalized Email"

    body = f"Dear {recipient['name']},\n\nThis is a personalized email just for you!"
    message.attach(MIMEText(body, "plain"))

    server = smtplib.SMTP("smtp.example.com", 587)
    server.starttls()
    server.login(sender_email, sender_password)
    server.send_message(message)
    server.quit()
