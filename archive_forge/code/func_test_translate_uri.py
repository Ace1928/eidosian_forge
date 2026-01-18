from pyxnat import uriutil
def test_translate_uri():
    assert uriutil.translate_uri('/assessors/out_resources/files') == '/assessors/out/resources/files'
    assert uriutil.translate_uri('/assessors/out_resource/files') == '/assessors/out/resource/files'
    assert uriutil.translate_uri('/assessors/in_resources/files') == '/assessors/in/resources/files'
    assert uriutil.translate_uri('/assessors/in_resource/files') == '/assessors/in/resource/files'