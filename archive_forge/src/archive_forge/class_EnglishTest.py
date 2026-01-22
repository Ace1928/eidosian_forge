import datetime
import os
import shutil
import tempfile
import unittest
import tornado.locale
from tornado.escape import utf8, to_unicode
from tornado.util import unicode_type
class EnglishTest(unittest.TestCase):

    def test_format_date(self):
        locale = tornado.locale.get('en_US')
        date = datetime.datetime(2013, 4, 28, 18, 35)
        self.assertEqual(locale.format_date(date, full_format=True), 'April 28, 2013 at 6:35 pm')
        aware_dt = datetime.datetime.now(datetime.timezone.utc)
        naive_dt = aware_dt.replace(tzinfo=None)
        for name, now in {'aware': aware_dt, 'naive': naive_dt}.items():
            with self.subTest(dt=name):
                self.assertEqual(locale.format_date(now - datetime.timedelta(seconds=2), full_format=False), '2 seconds ago')
                self.assertEqual(locale.format_date(now - datetime.timedelta(minutes=2), full_format=False), '2 minutes ago')
                self.assertEqual(locale.format_date(now - datetime.timedelta(hours=2), full_format=False), '2 hours ago')
                self.assertEqual(locale.format_date(now - datetime.timedelta(days=1), full_format=False, shorter=True), 'yesterday')
                date = now - datetime.timedelta(days=2)
                self.assertEqual(locale.format_date(date, full_format=False, shorter=True), locale._weekdays[date.weekday()])
                date = now - datetime.timedelta(days=300)
                self.assertEqual(locale.format_date(date, full_format=False, shorter=True), '%s %d' % (locale._months[date.month - 1], date.day))
                date = now - datetime.timedelta(days=500)
                self.assertEqual(locale.format_date(date, full_format=False, shorter=True), '%s %d, %d' % (locale._months[date.month - 1], date.day, date.year))

    def test_friendly_number(self):
        locale = tornado.locale.get('en_US')
        self.assertEqual(locale.friendly_number(1000000), '1,000,000')

    def test_list(self):
        locale = tornado.locale.get('en_US')
        self.assertEqual(locale.list([]), '')
        self.assertEqual(locale.list(['A']), 'A')
        self.assertEqual(locale.list(['A', 'B']), 'A and B')
        self.assertEqual(locale.list(['A', 'B', 'C']), 'A, B and C')

    def test_format_day(self):
        locale = tornado.locale.get('en_US')
        date = datetime.datetime(2013, 4, 28, 18, 35)
        self.assertEqual(locale.format_day(date=date, dow=True), 'Sunday, April 28')
        self.assertEqual(locale.format_day(date=date, dow=False), 'April 28')