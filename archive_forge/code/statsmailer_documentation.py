from scrapy import signals
from scrapy.exceptions import NotConfigured
from scrapy.mail import MailSender

StatsMailer extension sends an email when a spider finishes scraping.

Use STATSMAILER_RCPTS setting to enable and give the recipient mail address
