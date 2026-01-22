import warnings
def _make_viztracer_initializer_and_initargs():
    try:
        import viztracer
        tracer = viztracer.get_tracer()
        if tracer is not None and getattr(tracer, 'enable', False):
            return (_viztracer_init, (tracer.init_kwargs,))
    except ImportError:
        pass
    except Exception as e:
        warnings.warn(f'Unable to introspect viztracer state: {e}')
    return (None, ())