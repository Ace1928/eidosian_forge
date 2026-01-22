from __future__ import annotations
import abc
from argparse import Namespace
import configparser
import logging
import os
from pathlib import Path
import re
import sys
from typing import Any
from sqlalchemy.testing import asyncio
@post
def _engine_uri(options, file_config):
    from sqlalchemy import testing
    from sqlalchemy.testing import config
    from sqlalchemy.testing import provision
    from sqlalchemy.engine import url as sa_url
    if options.dburi:
        db_urls = list(options.dburi)
    else:
        db_urls = []
    extra_drivers = options.dbdriver or []
    if options.db:
        for db_token in options.db:
            for db in re.split('[,\\s]+', db_token):
                if db not in file_config.options('db'):
                    raise RuntimeError("Unknown URI specifier '%s'.  Specify --dbs for known uris." % db)
                else:
                    db_urls.append(file_config.get('db', db))
    if not db_urls:
        db_urls.append(file_config.get('db', 'default'))
    config._current = None
    if options.write_idents and provision.FOLLOWER_IDENT:
        for db_url in [sa_url.make_url(db_url) for db_url in db_urls]:
            with open(options.write_idents, 'a') as file_:
                file_.write(f'{provision.FOLLOWER_IDENT} {db_url.render_as_string(hide_password=False)}\n')
    expanded_urls = list(provision.generate_db_urls(db_urls, extra_drivers))
    for db_url in expanded_urls:
        log.info('Adding database URL: %s', db_url)
        cfg = provision.setup_config(db_url, options, file_config, provision.FOLLOWER_IDENT)
        if not config._current:
            cfg.set_as_current(cfg, testing)