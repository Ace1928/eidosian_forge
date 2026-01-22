from fontTools.cffLib import maxStackLimit
def _convertToBlendCmds(args):
    num_args = len(args)
    stack_use = 0
    new_args = []
    i = 0
    while i < num_args:
        arg = args[i]
        if not isinstance(arg, list):
            new_args.append(arg)
            i += 1
            stack_use += 1
        else:
            prev_stack_use = stack_use
            num_sources = len(arg) - 1
            blendlist = [arg]
            i += 1
            stack_use += 1 + num_sources
            while i < num_args and isinstance(args[i], list):
                blendlist.append(args[i])
                i += 1
                stack_use += num_sources
                if stack_use + num_sources > maxStackLimit:
                    break
            num_blends = len(blendlist)
            blend_args = []
            for arg in blendlist:
                blend_args.append(arg[0])
            for arg in blendlist:
                assert arg[-1] == 1
                blend_args.extend(arg[1:-1])
            blend_args.append(num_blends)
            new_args.append(blend_args)
            stack_use = prev_stack_use + num_blends
    return new_args