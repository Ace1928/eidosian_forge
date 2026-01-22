import gettext
import logging
import os
import sqlite3
import sys
class Country(object):

    def __init__(self, rowid):
        country = _database.execute('SELECT * FROM countries WHERE rowid == ?', (rowid,)).fetchone()
        self.name = country[0]
        self.official_name = country[1]
        self.alpha_2 = country[2]
        self.alpha_3 = country[3]
        self.numeric = country[4]
        self.translation = _translator_country(self.name)

    @classmethod
    def get_country(cls, code, codec):
        country = _database.execute('SELECT rowid FROM countries WHERE %s == ?' % codec, (code,)).fetchone()
        if country:
            return cls(country[0])
        raise CountryNotFound('code: %s, codec: %s' % (code, codec))

    @classmethod
    def by_alpha_2(cls, code):
        return Country.get_country(code, 'alpha_2')

    @classmethod
    def by_alpha_3(cls, code):
        return Country.get_country(code, 'alpha_3')

    @classmethod
    def by_numeric(cls, code):
        return Country.get_country(code, 'numeric')