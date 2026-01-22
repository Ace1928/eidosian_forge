from functools import partial
from unittest import mock
import betamax
import fixtures
import requests
from keystoneauth1.fixture import hooks
from keystoneauth1.fixture import serializer as yaml_serializer
from keystoneauth1 import session
def _construct_session_with_betamax(fixture, session_obj=None):
    if not session_obj:
        session_obj = requests.Session()
        for scheme in list(session_obj.adapters.keys()):
            session_obj.mount(scheme, session.TCPKeepAliveAdapter())
    with betamax.Betamax.configure() as config:
        config.before_record(callback=fixture.pre_record_hook)
    fixture.recorder = betamax.Betamax(session_obj, cassette_library_dir=fixture.cassette_library_dir)
    record = 'none'
    serializer = None
    if fixture.record in ['once', 'all', 'new_episodes']:
        record = fixture.record
    serializer = fixture.serializer_name
    fixture.recorder.use_cassette(fixture.cassette_name, serialize_with=serializer, record=record, **fixture.use_cassette_kwargs)
    fixture.recorder.start()
    fixture.addCleanup(fixture.recorder.stop)
    return session_obj