def __lldb_init_module(debugger, internal_dict):
    import lldb
    debugger.HandleCommand('command script add -f lldb_prepare.load_lib_and_attach load_lib_and_attach')
    try:
        target = debugger.GetSelectedTarget()
        if target:
            process = target.GetProcess()
            if process:
                for thread in process:
                    internal_dict['_thread_%d' % thread.GetThreadID()] = True
    except:
        import traceback
        traceback.print_exc()